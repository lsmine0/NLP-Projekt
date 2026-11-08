
import handlers.ContextHandle as Contx
import Logger
import json


class AgentFlow:

    def __init__(self, ollama_handler, max_iterations=10):

        self.ollama = ollama_handler
        self.tools = ollama_handler.ToolsHandle

        self.context = []
        self.max_iterations = max_iterations

    def _AddToolCalls(self, message):

        Contx.ContextHandle.Append(
            self.context,
            message
        )

    def _ExecuteToolCalls(self, tool_calls):

        for tool_call in tool_calls:


            name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            Logger.printInfo(f"Executing tool: {name}")

            result = self.tools.execute_tool(
                name,
                arguments
            )

            Contx.ContextHandle.Add(
                self.context,
                result,
                role="tool",
                tool_call_id=tool_call.id
            )   

    def Run(self, user_input):

        Contx.ContextHandle.Add(
            self.context,
            user_input,
            role="user"
        )

        for iteration in range(self.max_iterations):

            Logger.printInfo(
                f"Agent iteration: {iteration + 1}"
            )

            response = self.ollama.Chat(
                messages=self.context,
            )

            message = response.choices[0].message

            if message.reasoning_content:
                Logger.printThinking(message.reasoning_content)

            if not message.tool_calls:

                answer = message.content or ""

                Contx.ContextHandle.Add(
                    self.context,
                    answer,
                    role="assistant"
                )

                return answer

            # Preserve assistant tool calls.
            self._AddToolCalls(
                message.model_dump(exclude_none=True)
            )

            # Execute requested tools.
            self._ExecuteToolCalls(message.tool_calls)

        Logger.printWarning(
            "Maximum agent iterations reached."
        )

        return None

    def RunStream(self, user_input):

        Contx.ContextHandle.Add(
            self.context,
            user_input,
            role="user"
        )

        for iteration in range(self.max_iterations):

            Logger.printInfo(
                f"Agent iteration: {iteration + 1}"
            )

            response = self.ollama.ChatStream(
                messages=self.context,
            )

            content = ""
            tool_calls = {}
            thinking_started = False
            # Collect streamed response.
            for chunk in response:

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                reasoning = (
                    getattr(delta, "reasoning_content", None)
                    or getattr(delta, "reasoning", None)
                    or getattr(delta, "thinking", None)
                )

                if reasoning:
                    Logger.printThinkingStream(reasoning, start=not thinking_started)
                    thinking_started = True

                # Stream text immediately.
                if delta.content:

                    content += delta.content

                    yield delta.content

                # Collect fragmented tool calls.
                if delta.tool_calls:

                    for tc in delta.tool_calls:

                        index = tc.index

                        if index not in tool_calls:

                            tool_calls[index] = {
                                "id": "",
                                "type": "function",
                                "function": {
                                    "name": "",
                                    "arguments": ""
                                }
                            }

                        current = tool_calls[index]

                        if tc.id:
                            current["id"] = tc.id

                        if tc.function:

                            if tc.function.name:
                                current["function"]["name"] += (
                                    tc.function.name
                                )

                            if tc.function.arguments:
                                current["function"]["arguments"] += (
                                    tc.function.arguments
                                )

            if not tool_calls:

                Contx.ContextHandle.Add(
                    self.context,
                    content,
                    role="assistant"
                )

                return

            # Reconstruct assistant tool-call message.
            assistant_message = {
                "role": "assistant",
                "content": content or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": tc["function"]
                    }
                    for tc in tool_calls.values()
                ]
            }

            self._AddToolCalls(assistant_message)

            # Execute tools.
            from types import SimpleNamespace

            calls = [
                SimpleNamespace(
                    id=tc["id"],
                    function=SimpleNamespace(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"]
                    )
                )
                for tc in tool_calls.values()
            ]

            self._ExecuteToolCalls(calls)

        Logger.printWarning(
            "Maximum agent iterations reached."
        )
