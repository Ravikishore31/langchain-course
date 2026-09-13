from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

load_dotenv()  # Load environment variables from .env file

MAX_ITERATIONS = 10
model = "gemma4:31b-cloud"


@tool
def get_product_price(product: str) -> float:
    """Look up the price of the product in catelog"""

    print(f"Looking up price for product: {product}")

    prices = {"laptop": 999.99, "smartphone": 699.99, "headphones": 199.99}
    return prices.get(product, 0.0)  # Return 0.0 if product not found


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount based on the discount tier
    Available discount tiers: silver, gold, platinum"""

    print(f"Applying {discount_tier} discount to price: {price}")

    discounts = {"silver": 5, "gold": 10, "platinum": 15}
    discount = discounts.get(discount_tier, 0)
    return round(
        price * (1 - discount / 100), 2
    )  # Return discounted price rounded to 2 decimal places


@traceable(name="React Under the Hood Agent")
def run_agent(question: str):

    tools = [get_product_price, apply_discount]

    tool_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(f"google_genai:gemini-3.6-flash", temperature=0)

    tool_bind = llm.bind_tools(tools)

    print(f"question: {question}   ")

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):

        print(f"\n-- Iteration {iteration} -- ")

        ai_message = tool_bind.invoke(messages)

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\n Final ANSWER: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tool_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"  [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello Langchain Community!!")
    print()
    result = run_agent("what is the price of a laptop with a gold discount?")
