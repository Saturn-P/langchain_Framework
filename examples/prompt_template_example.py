"""
LangChain PromptTemplate and LLMChain Example
==============================================

This example demonstrates how to use LangChain's PromptTemplate with an LLM chain.
It's beginner-friendly and requires minimal setup.

Requirements:
    pip install langchain openai

Setup:
    Set your OpenAI API key as an environment variable:
    - Windows: set OPENAI_API_KEY=your-api-key-here
    - Linux/Mac: export OPENAI_API_KEY=your-api-key-here
    
    Or create a .env file with:
    OPENAI_API_KEY=your-api-key-here
"""

import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI


def main():
    """
    Main function demonstrating PromptTemplate and LLMChain usage.
    """
    
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it before running this example."
        )
    
    print("=" * 60)
    print("LangChain PromptTemplate & LLMChain Example")
    print("=" * 60)
    print()
    
    # -------------------------------------------------------------------------
    # Example 1: Simple PromptTemplate with single input variable
    # -------------------------------------------------------------------------
    print("Example 1: Simple Question Answering")
    print("-" * 60)
    
    # Define a prompt template with one input variable: {topic}
    simple_template = """You are a helpful assistant. 
    
Question: What is {topic}?

Answer: Let me explain {topic} in simple terms."""
    
    # Create the PromptTemplate object
    simple_prompt = PromptTemplate(
        input_variables=["topic"],
        template=simple_template
    )
    
    # Initialize the OpenAI LLM
    # temperature=0.7 controls creativity (0=deterministic, 1=creative)
    llm = OpenAI(temperature=0.7)
    
    # Create an LLMChain by combining the prompt and LLM
    simple_chain = LLMChain(llm=llm, prompt=simple_prompt)
    
    # Run the chain with input
    topic = "photosynthesis"
    result = simple_chain.run(topic=topic)
    
    print(f"Topic: {topic}")
    print(f"Response: {result}")
    print()
    
    # -------------------------------------------------------------------------
    # Example 2: PromptTemplate with multiple input variables
    # -------------------------------------------------------------------------
    print("Example 2: Multiple Input Variables")
    print("-" * 60)
    
    # Create a template with multiple variables: {product}, {audience}, {tone}
    multi_template = """You are a marketing copywriter.

Write a {tone} product description for {product} targeting {audience}.

Product Description:"""
    
    multi_prompt = PromptTemplate(
        input_variables=["product", "audience", "tone"],
        template=multi_template
    )
    
    # Create another LLMChain with the new prompt
    multi_chain = LLMChain(llm=llm, prompt=multi_prompt)
    
    # Run with multiple inputs
    result = multi_chain.run(
        product="eco-friendly water bottle",
        audience="environmentally conscious millennials",
        tone="enthusiastic and inspiring"
    )
    
    print(f"Product: eco-friendly water bottle")
    print(f"Audience: environmentally conscious millennials")
    print(f"Tone: enthusiastic and inspiring")
    print(f"Response: {result}")
    print()
    
    # -------------------------------------------------------------------------
    # Example 3: Using format() method directly on PromptTemplate
    # -------------------------------------------------------------------------
    print("Example 3: Direct Prompt Formatting")
    print("-" * 60)
    
    # Create a template for code explanation
    code_template = PromptTemplate(
        input_variables=["language", "code"],
        template="Explain the following {language} code:\n\n{code}\n\nExplanation:"
    )
    
    # Format the prompt without running the chain
    formatted_prompt = code_template.format(
        language="Python",
        code="result = [x**2 for x in range(10)]"
    )
    
    print("Formatted Prompt:")
    print(formatted_prompt)
    print()
    
    # Now run it through the LLM
    code_chain = LLMChain(llm=llm, prompt=code_template)
    result = code_chain.run(
        language="Python",
        code="result = [x**2 for x in range(10)]"
    )
    
    print(f"Response: {result}")
    print()
    
    # -------------------------------------------------------------------------
    # Example 4: Using template from file or string with partial variables
    # -------------------------------------------------------------------------
    print("Example 4: Partial Variables")
    print("-" * 60)
    
    # Create a template with partial variables pre-filled
    partial_template = PromptTemplate(
        input_variables=["question"],
        template="You are an expert in {domain}. Answer this question: {question}",
        partial_variables={"domain": "artificial intelligence"}
    )
    
    partial_chain = LLMChain(llm=llm, prompt=partial_template)
    
    # Only need to provide the 'question' variable
    result = partial_chain.run(question="What is machine learning?")
    
    print(f"Question: What is machine learning?")
    print(f"Response: {result}")
    print()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Installed required packages: pip install langchain openai")
        print("2. Set OPENAI_API_KEY environment variable")
