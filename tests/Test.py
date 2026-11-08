
from OllamaHandle import OllamaHandler
from AgentFlow import AgentLoop


def AddNumbers(a: float, b: float):
    return a + b


def main():

    # Initialize Ollama.
    ollama = OllamaHandler()

    models = [
    model.id
    for model in ollama.GetModels().data
    ]

    if not models:
        print("No models available.")
        return

    # Main-side model selection.
    print("\nAvailable models:")

    for index, model in enumerate(models):
        print(f"[{index}] {model}")

    while True:

        selection = input("\nSelect model (number): ").strip()

        try:
            index = int(selection)

            if 0 <= index < len(models):
                break

            print("Invalid selection.")

        except ValueError:
            print("Please enter a number.")

    selected_model = models[index]

    if not ollama.SetModel(selected_model):
        print("Failed to select model.")
        return

    print(f"Selected model: {selected_model}")


    # Register tool schema.
    ollama.ToolsHandle.create_tool(
        name="add_numbers",
        description="Adds two numbers together.",
        properties={
            "a": {
                "type": "number",
                "description": "First number"
            },
            "b": {
                "type": "number",
                "description": "Second number"
            }
        },
        required=["a", "b"]
    )

    # Register Python implementation.
    ollama.ToolsHandle.register_function(
        "add_numbers",
        AddNumbers
    )

    # Initialize agent.
    agent = AgentLoop(ollama)

    # Select execution mode.
    stream = input("Enable streaming? [y/N]: ").lower() == "y"

    print("\nAgent ready. Type 'exit' to quit.\n")

    while True:

        try:

            user_input = input("You: ").strip()

            if user_input.lower() in ("exit", "quit"):
                break

            if not user_input:
                continue

            print("Assistant: ", end="", flush=True)

            if stream:

                for chunk in agent.RunStream(user_input):
                    print(chunk, end="", flush=True)

                print()

            else:

                answer = agent.Run(user_input)

                print(answer)

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break

        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
