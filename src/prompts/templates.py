"""
Prompt templates for ResearchRAG.

This module centralizes all prompt engineering for the application. It defines
reusable templates for various retrieval-augmented generation strategies,
ensuring that prompts remain separate from business logic, are easily
testable, and that the system interacts with language models consistently.
"""

from dataclasses import dataclass
from typing import Any, List

from src.config.logging_config import get_logger

logger = get_logger(__name__)


class PromptTemplateError(Exception):
    """Base exception for prompt template formatting errors."""


@dataclass(frozen=True)
class PromptTemplate:
    """
    Represents a reusable prompt template.

    Provides a standardized way to format prompts with required variables
    and validates that all necessary variables are provided before passing
    the prompt to the language model.
    """

    name: str
    template: str
    required_variables: List[str]

    def format(self, **kwargs: Any) -> str:
        """
        Format the template with the provided variables.

        Args:
            **kwargs: Keyword arguments matching the required variables.

        Returns:
            The fully formatted prompt string ready for LLM consumption.

        Raises:
            PromptTemplateError: If any required variables are missing
                or if formatting fails.
        """
        missing_vars = [var for var in self.required_variables if var not in kwargs]
        if missing_vars:
            raise PromptTemplateError(
                f"Cannot format template '{self.name}': missing required "
                f"variables: {missing_vars}"
            )

        try:
            return self.template.format(**kwargs)
        except KeyError as error:
            raise PromptTemplateError(
                f"Template '{self.name}' expects variable '{error.args[0]}' "
                f"but it was not provided in kwargs."
            ) from error
        except Exception as error:
            raise PromptTemplateError(
                f"Error formatting template '{self.name}': {error}"
            ) from error


class RAGPrompts:
    """
    A centralized library of standardized RAG prompt templates.

    These templates dictate how the language model should behave,
    enforce anti-hallucination rules, format context, and structure
    the final output. The RAG Pipeline should retrieve templates
    from here rather than hardcoding strings.
    """

    # 1. Basic RAG Question Answering
    BASIC_QA = PromptTemplate(
        name="basic_qa",
        template=(
            "You are an expert research assistant. Answer the user's question "
            "based ONLY on the provided context.\n\n"
            "Context:\n"
            "{context}\n\n"
            "Question: {user_question}\n\n"
            "Answer:"
        ),
        required_variables=["context", "user_question"],
    )

    # 2. Strict Grounded Answering
    STRICT_GROUNDED = PromptTemplate(
        name="strict_grounded",
        template=(
            "You are a strict, factual research assistant. You must answer the "
            "user's question using EXCLUSIVELY the information found in the "
            "provided context.\n"
            "- Do NOT use prior knowledge.\n"
            "- Do NOT hallucinate or infer information not explicitly stated.\n"
            "- If the context does not contain enough information to answer the "
            "question, you MUST reply exactly with: 'I don't know based on the "
            "provided context.'\n\n"
            "Context:\n"
            "{context}\n\n"
            "Question: {user_question}\n\n"
            "Answer:"
        ),
        required_variables=["context", "user_question"],
    )

    # 3. Citation-Based Answering
    CITATION_QA = PromptTemplate(
        name="citation_qa",
        template=(
            "You are an academic research assistant. Answer the user's question "
            "based on the provided context. When you use information from the "
            "context, you MUST append a citation to the end of the sentence "
            "using the document's source identifier (e.g., [Document 1]).\n\n"
            "Context:\n"
            "{context}\n\n"
            "Question: {user_question}\n\n"
            "Answer with citations:"
        ),
        required_variables=["context", "user_question"],
    )

    # 4. Summarization
    SUMMARIZATION = PromptTemplate(
        name="summarization",
        template=(
            "Summarize the following retrieved documents. Capture the main ideas, "
            "key arguments, and any significant conclusions. Keep the summary "
            "concise but comprehensive, preserving technical accuracy.\n\n"
            "Documents:\n"
            "{context}\n\n"
            "Summary:"
        ),
        required_variables=["context"],
    )

    # 5. Document Explanation
    DOCUMENT_EXPLANATION = PromptTemplate(
        name="document_explanation",
        template=(
            "Explain the concepts discussed in the following context to someone "
            "who is not an expert in the field. Break down complex jargon into "
            "simple terms, but strictly preserve the technical accuracy of the "
            "original text.\n\n"
            "Context:\n"
            "{context}\n\n"
            "Explanation:"
        ),
        required_variables=["context"],
    )

    # 6. Follow-up Question Answering
    FOLLOW_UP_QA = PromptTemplate(
        name="follow_up_qa",
        template=(
            "You are an expert research assistant. Answer the user's follow-up "
            "question based on the provided context and the conversation history.\n\n"
            "Conversation History:\n"
            "{conversation_history}\n\n"
            "Context:\n"
            "{context}\n\n"
            "Follow-up Question: {user_question}\n\n"
            "Answer:"
        ),
        required_variables=["context", "user_question", "conversation_history"],
    )

    # 7. Context-Aware Question Answering
    CONTEXT_AWARE = PromptTemplate(
        name="context_aware",
        template=(
            "You are a helpful research assistant. Analyze the provided context "
            "and answer the user's question. If the context partially answers "
            "the question, provide the partial answer and clearly state what "
            "information is missing. Do not guess or hallucinate the missing "
            "information.\n\n"
            "Context:\n"
            "{context}\n\n"
            "Question: {user_question}\n\n"
            "Answer:"
        ),
        required_variables=["context", "user_question"],
    )

    # 8. "I Don't Know" Fallback Behavior
    FALLBACK_NO_CONTEXT = PromptTemplate(
        name="fallback_no_context",
        template=(
            "The system was unable to retrieve any relevant documents for the "
            "following query. Acknowledge the user's question and politely "
            "inform them that you do not have the required context to provide "
            "a factual answer. Do not attempt to answer the question.\n\n"
            "Question: {user_question}\n\n"
            "Response:"
        ),
        required_variables=["user_question"],
    )
