# custom_agent.py
import warnings
from typing import Any, Dict, List, Literal, Optional, Sequence, Union, cast

from langchain.agents import AgentExecutor, AgentType
from langchain.agents.agent import BaseSingleActionAgent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.callbacks.base import BaseCallbackManager
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain.schema import SystemMessage
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_core.tools.base import BaseTool
from langchain_core.utils.interactive_env import is_interactive_env

# Prompt templates
PREFIX = """
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
You should use the tools below to answer the question posed to you:
"""

SUFFIX_WITH_DF = """
This is the result of `print(df.head())`:
{df_head}

This is the result of `print(df.describe())`:
{df_describe}

Begin!
Question: {{question}}
{{agent_scratchpad}}"""

SUFFIX_NO_DF = """
Begin!
Question: {{question}}
{{agent_scratchpad}}"""

def custom_create_pandas_dataframe_agent(
    llm: BaseChatModel,
    df: Any,
    agent_type: Union[AgentType, Literal["zero-shot-react-description"]] = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
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
    allow_dangerous_code: bool = True,
    **kwargs: Any,
) -> (PromptTemplate, AgentExecutor):
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

    # Prepare tools
    df_locals = {"df": df}
    tools = [PythonAstREPLTool(locals=df_locals)] + list(extra_tools)

    # Prepare prompt
    if include_df_in_prompt:
        df_head = df.head(number_of_head_rows).to_markdown()
        df_describe = df.describe().to_markdown()
        formatted_suffix = (suffix or SUFFIX_WITH_DF).format(
            df_head=df_head,
            df_describe=df_describe,
        )
    else:
        formatted_suffix = suffix or SUFFIX_NO_DF

    prompt_template = (prefix or PREFIX) + "\n\n" + formatted_suffix

    prompt = PromptTemplate(
        input_variables=["question", "agent_scratchpad"],
        template=prompt_template,
    )

    # Create agent
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        agent_type=agent_type,
        callback_manager=callback_manager,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        agent_executor_kwargs=agent_executor_kwargs,
        tools=tools,
        prompt=prompt,
        **kwargs,
    )

    return prompt, agent