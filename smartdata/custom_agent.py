# custom_agent.py
import warnings
from typing import Any, Dict, List, Literal, Optional, Sequence, Union, cast

from langchain.agents import (
    AgentType,
    create_openai_tools_agent,
    create_react_agent,
    create_tool_calling_agent,
)
from langchain.agents.agent import (
    AgentExecutor,
    BaseMultiActionAgent,
    BaseSingleActionAgent,
    RunnableAgent,
    RunnableMultiActionAgent,
)
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.agents.openai_functions_agent.base import (
    OpenAIFunctionsAgent,
    create_openai_functions_agent,
)
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel, LanguageModelLike
from langchain_core.messages import SystemMessage
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.utils.interactive_env import is_interactive_env

from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")

# Prompt templates
PREFIX = """
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
You should use the tools below to answer the question posed to you:
"""

MULTI_DF_PREFIX = """
You are working with {num_dfs} pandas dataframes in Python named df1, df2, etc.
You should use the tools below to answer the question posed to you:
"""

SUFFIX_NO_DF = """
Begin!
Question: {input}
{agent_scratchpad}"""

SUFFIX_WITH_DF = """
This is the result of `print(df.head())`:
{df_head}

This is the result of `print(df.describe())`:
{df_describe}

Begin!
Question: {input}
{agent_scratchpad}"""

SUFFIX_WITH_MULTI_DF = """
This is the result of `print(df.head())` for each dataframe:
{dfs_head}

This is the result of `print(df.describe())` for each dataframe:
{dfs_describe}

Begin!
Question: {input}
{agent_scratchpad}"""

PREFIX_FUNCTIONS = """
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
"""

MULTI_DF_PREFIX_FUNCTIONS = """
You are working with {num_dfs} pandas dataframes in Python named df1, df2, etc.
"""

FUNCTIONS_WITH_DF = """
This is the result of `print(df.head())`:
{df_head}

This is the result of `print(df.describe())`:
{df_describe}

This is the result of `print(df.dtypes)`:
{df_dtypes}

This is the result of df.value_counts for each non-numeric column in a dictionary (limit to the first 10 if more than 10 unique value counts):
{df_col_unique_value_counts}
"""

FUNCTIONS_WITH_MULTI_DF = """
This is the result of `print(df.head())` for each dataframe:
{dfs_head}

This is the result of `print(df.describe())` for each dataframe:
{dfs_describe}
"""

def _get_df_col_value_counts(df):
    """Get the top 10 value counts for each categorical column."""
    df_checking = df.copy()
    boolean_and_datetime_columns = df_checking.select_dtypes(include=['boolean', 'datetime64', 'Interval']).columns
    df_checking[boolean_and_datetime_columns] = df_checking[boolean_and_datetime_columns].astype(str)
    categorical_columns = df_checking.select_dtypes(include=['object', 'category', 'string']).columns
    top_10_values = df_checking[categorical_columns].apply(
        lambda col: col.value_counts(dropna=False).head(10).to_dict()
    ).to_dict()
    return str(top_10_values)

def _get_functions_single_prompt(
    df: Any,
    prefix: Optional[str] = None,
    suffix: str = "",
    include_df_in_prompt: bool = True,
    number_of_head_rows: int = 5,
) -> BasePromptTemplate:
    if include_df_in_prompt:
        df_head = df.head(number_of_head_rows).to_markdown()
        df_describe = df.describe().to_markdown()
        df_dtypes = df.dtypes.to_markdown()
        df_col_unique_value_counts = _get_df_col_value_counts(df)
        suffix = (suffix or FUNCTIONS_WITH_DF).format(
            df_head=df_head,
            df_describe=df_describe,
            df_dtypes=df_dtypes,
            df_col_unique_value_counts=df_col_unique_value_counts
        )
    prefix = prefix or PREFIX_FUNCTIONS
    system_message = SystemMessage(content=prefix + suffix)
    prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
    return prompt

def custom_create_pandas_dataframe_agent(
    llm: LanguageModelLike,
    df: Any,
    agent_type: Union[AgentType, Literal["openai-tools", "tool-calling"]] = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    verbose: bool = False,
    return_intermediate_steps: bool = False,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    include_df_in_prompt: bool = True,
    number_of_head_rows: int = 5,
    extra_tools: Sequence[BaseTool] = (),
    engine: Literal["pandas", "modin"] = "pandas",
    allow_dangerous_code: bool = False,
    **kwargs: Any,
) -> AgentExecutor:
    """
    Construct a Pandas agent from an LLM and dataframe(s).
    """
    if not allow_dangerous_code:
        raise ValueError(
            "This agent relies on access to a python repl tool which can execute "
            "arbitrary code. You must opt-in to use this functionality by setting "
            "allow_dangerous_code=True."
        )

    # Import pandas or modin
    try:
        if engine == "modin":
            import modin.pandas as pd
        elif engine == "pandas":
            import pandas as pd
        else:
            raise ValueError(
                f"Unsupported engine {engine}. It must be one of 'modin' or 'pandas'."
            )
    except ImportError as e:
        raise ImportError(
            f"`{engine}` package not found, please install with `pip install {engine}`"
        ) from e

    if is_interactive_env():
        pd.set_option("display.max_columns", None)

    # Validate dataframe(s)
    df_list = df if isinstance(df, list) else [df]
    for _df in df_list:
        if not isinstance(_df, pd.DataFrame):
            raise ValueError(f"Expected pandas DataFrame, got {type(_df)}")

    # Prepare tools
    df_locals = {f"df{i+1}": dataframe for i, dataframe in enumerate(df_list)} if isinstance(df, list) else {"df": df}
    tools = [PythonAstREPLTool(locals=df_locals)] + list(extra_tools)

    # Prepare prompt
    prompt = _get_functions_single_prompt(
        df,
        prefix=prefix,
        suffix=suffix,
        include_df_in_prompt=include_df_in_prompt,
        number_of_head_rows=number_of_head_rows,
    )

    # Create agent
    runnable = create_openai_functions_agent(cast(BaseLanguageModel, llm), tools, prompt)
    agent = RunnableAgent(
        runnable=runnable,
        input_keys_arg=["input"],
        return_keys_arg=["output"],
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        **(agent_executor_kwargs or {}),
    )
