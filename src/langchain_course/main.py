from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)


def main():

    information = """
    Samuel Harris Altman (born April 22, 1985) is an American entrepreneur and investor who has been the chief executive officer (CEO) of the artificial intelligence company OpenAI since 2019.

Altman attended Stanford University for two years before he dropped out and co-founded Loopt, a geosocial networking application for smartphones. Loopt was acquired by Green Dot Corporation for $43.4 million in March 2012.[1] In 2011, Altman joined Y Combinator, a technology startup accelerator and venture capital firm, and was the company's president from 2014 to 2019.[2] He is a billionaire, due to investments in over 400 companies including Reddit, Worldcoin, Helion Energy and Instacart.

Altman co-founded OpenAI in 2015 and became its CEO in 2019, a role that made him a prominent figure of the AI boom. He supervised the launch of ChatGPT in November 2022. In 2023, he was ousted by the organization's board of directors for not being "consistently candid".[3][4] The move was met with significant backlash from employees and investors, resulting in Altman's reinstatement five days later and the formation of a new board.
    """

    summarize_prompt = """

    Given the information {information}, please provide a concise summary of the key points, highlighting the most important details and insights. The summary should be clear, coherent, and capture the essence of the information provided."""

    prompt_template = PromptTemplate(
        input_variables=["information"], template=summarize_prompt
    )

    llm = ChatGoogleGenerativeAI(
        temperature=0,
        model="gemini-3.8-flash",
    ).bind(automatic_function_calling={"disable": True})
    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke(input={"information": information})
    print(response)


if __name__ == "__main__":
    main()
