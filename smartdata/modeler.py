# modeler.py
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import pandas as pd
import copy
import logging
import time
from functools import wraps

from .config import Config
from .memory import Memory
from .custom_agent import custom_create_pandas_dataframe_agent
from .util import clean_dataframe

# Import exceptions for rate limiting
from openai import RateLimitError, Timeout

logger = logging.getLogger('SmartData')

def retry_on_rate_limit(max_retries=5, initial_delay=1, backoff_factor=2):
    """
    Decorator to retry a function if RateLimitError or Timeout is encountered.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (RateLimitError, Timeout) as e:
                    retries += 1
                    logger.warning(f"Rate limit or timeout encountered. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= backoff_factor
            raise Exception("Maximum retries exceeded due to rate limiting.")
        return wrapper
    return decorator

class SmartData:
    def __init__(self, df_list, llm=None, show_detail=Config.SHOW_DETAIL, memory_size=Config.MEMORY_SIZE,
                 max_iterations=Config.MAX_ITERATIONS, max_execution_time=Config.MAX_EXECUTION_TIME, seed=0):

        self.llm = llm or ChatOpenAI(temperature=Config.TEMP_CHAT, model=Config.CHAT_MODEL, seed=seed)
        self.df_list = copy.deepcopy(df_list)
        self.df_change = []
        self.memory_size = memory_size
        self.max_iterations = max_iterations
        self.max_execution_time = max_execution_time
        self.show_detail = show_detail
        self.image_fig_list = []
        self.check_error_substring_list = Config.CHECK_ERROR_SUBSTRING_LIST
        self.check_plot_substring_list = Config.CHECK_PLOT_SUBSTRING_LIST
        self.add_on_plot_library_list = Config.ADD_ON_PLOT_LIBRARY_LIST
        self.check_datachange_substring_list = Config.CHECK_DATACHANGE_SUBSTRING_LIST
        self.add_on_datachange_library_list = Config.ADD_ON_DATACHANGE_LIBRARY_LIST
        self.prompt_clean_data = Config.PROMPT_CLEAN_DATA
        self.prompt_create_data_clean_summary = Config.PROMPT_CREATE_DATA_CLEAN_SUMMARY
        self.model = None
        self.memory = Memory()
        self.message_count = 1

    def create_model(self, use_openai_llm=True, seed=0):
        df = self.df_list
        prefix_df = Config.DEFAULT_PREFIX_SINGLE_DF
        if use_openai_llm:
            self.llm = ChatOpenAI(temperature=Config.TEMP_CHAT, model=Config.CHAT_MODEL, seed=seed)

        prompt, agent_executor = custom_create_pandas_dataframe_agent(
            llm=self.llm,
            df=df,
            verbose=self.show_detail,
            return_intermediate_steps=True,
            agent_type="zero-shot-react-description",
            allow_dangerous_code=True,
            prefix=prefix_df,
            max_iterations=self.max_iterations,
            max_execution_time=self.max_execution_time,
            agent_executor_kwargs={'handle_parsing_errors': True}
        )

        self.model = agent_executor
        return prompt, agent_executor

    @retry_on_rate_limit()
    def _invoke_model(self, question_with_history):
        """
        Invokes the model with retry logic in case of rate limiting.
        """
        return self.model.run(question=question_with_history)

    def run_model(self, question):
        answer = ""
        code_list_plot_with_add_on = []
        code_list_datachange_with_add_on = []
        response = None
        code_list = []
        for i in range(10):
            self.create_model(use_openai_llm=True, seed=i)
            try:
                self.image_fig_list.clear()
                self.df_change.clear()
                has_plots = False
                has_changes_to_df = False

                question_with_history = question
                if self.memory.is_not_empty():
                    last_conversations = self.memory.recall_last_conversation(self.memory_size)
                    question_with_history = (
                        f"My question is: {question}. Below is our previous conversation and codes "
                        f"in chronological order, from the earliest to the latest: {last_conversations}."
                    )

                # Use the retryable invoke method
                response = self._invoke_model(question_with_history)
                answer = response['output']
                code_list = self.extract_code_from_response(response)

                # Process plot code
                code_list_plot_with_add_on = self.process_plot_code(code_list)
                for plot_code in code_list_plot_with_add_on:
                    exec(plot_code, {'image_fig_list': self.image_fig_list, 'df': self.df_list}, {})
                if self.image_fig_list:
                    has_plots = True

                # Process data change code
                code_list_datachange_with_add_on = self.process_datachange_code(code_list)
                for data_code in code_list_datachange_with_add_on:
                    exec(data_code, {'df_change': self.df_change, 'df': self.df_list}, {})
                if self.df_change:
                    has_changes_to_df = not self.df_list.equals(self.df_change[-1])
                    self.df_list = copy.deepcopy(self.df_change[-1])
                    self.create_model(use_openai_llm=True, seed=i)

                # Store the chat history
                self.remember_conversation(question, answer, code_list)
                if any(substring in str(answer) for substring in Config.AGENT_STOP_SUBSTRING_LIST):
                    answer = Config.AGENT_STOP_ANSWER
                else:
                    break
            except Exception as e:
                logger.error(f"Failed to process: {e}")

        return answer, has_plots, has_changes_to_df, self.image_fig_list, self.df_list, response, code_list, code_list_plot_with_add_on, code_list_datachange_with_add_on

    @retry_on_rate_limit()
    def _invoke_summary_chain(self, result):
        """
        Invokes the summary chain with retry logic in case of rate limiting.
        """
        summary_prompt = PromptTemplate(
            template=self.prompt_create_data_clean_summary,
            input_variables=['result']
        )
        summary_model = ChatOpenAI(temperature=Config.TEMP_CHAT, model=Config.CHAT_MODEL, seed=0)
        chain = summary_prompt | summary_model | StrOutputParser()
        return chain.invoke({"result": result})

    def create_data_clean_summary(self, result):
        """
        Creates a summary of data cleaning results with retry logic.
        """
        try:
            answer = self._invoke_summary_chain(result)
            return self.prompt_create_data_clean_summary, answer
        except Exception as e:
            logger.error(f"Failed to create data clean summary: {e}")
            return self.prompt_create_data_clean_summary, ""

    def clean_data_without_ai(self):
        df_clean, summary = clean_dataframe(self.df_list)
        self.df_list = df_clean
        return summary, df_clean

    def clean_data_with_ai(self):
        answer, has_changes_to_df, *_ = self.run_model(question=self.prompt_clean_data)
        return answer, has_changes_to_df, self.df_list

    def clean_data(self):
        summary_without_ai, *_ = self.clean_data_without_ai()
        answer, has_changes_to_df, *_ = self.clean_data_with_ai()
        summary = summary_without_ai + answer
        _, final_summary = self.create_data_clean_summary(summary)
        return final_summary, has_changes_to_df, self.df_list

    def remember_conversation(self, question, answer, code_list):
        self.memory.remember(self.message_count, 'Human', question)
        self.memory.remember(self.message_count, 'AI', answer)
        self.memory.remember(self.message_count, 'Plot Code Generated By AI', code_list)
        self.message_count += 1

    def extract_code_from_response(self, response):
        code_list = []
        try:
            for action in response["intermediate_steps"]:
                if action[0].tool == "python_repl_ast":
                    code = action[0].tool_input
                    code_list.append(code)
        except Exception as e:
            logger.error(f"Error extracting code from response: {e}")
        return code_list

    def process_plot_code(self, code_list):
        code_list_plot = [
            code for code in code_list if all(sub in code for sub in self.check_plot_substring_list)
        ]
        code_list_plot = list(dict.fromkeys(code_list_plot))
        code_list_plot_with_add_on = []
        for code in code_list_plot:
            missing_imports = [
                lib for lib in self.add_on_plot_library_list if lib not in code
            ]
            code_with_imports = "\n".join(missing_imports) + "\n" + code if missing_imports else code
            code_with_format_label = code_with_imports + Config.ADD_ON_FORMAT_LABEL_FOR_AXIS
            code_with_fig = code_with_format_label + Config.ADD_ON_FIG
            code_list_plot_with_add_on.append(code_with_fig)
        return code_list_plot_with_add_on

    def process_datachange_code(self, code_list):
        code_list_datachange = [
            code for code in code_list if all(sub in code for sub in self.check_datachange_substring_list)
        ]
        code_list_datachange = list(dict.fromkeys(code_list_datachange))
        code_list_datachange_with_add_on = []
        for code in code_list_datachange:
            missing_imports = [
                lib for lib in self.add_on_datachange_library_list if lib not in code
            ]
            code_with_imports = "\n".join(missing_imports) + "\n" + code if missing_imports else code
            code_with_df = code_with_imports + Config.ADD_ON_DF
            code_list_datachange_with_add_on.append(code_with_df)
        return code_list_datachange_with_add_on
