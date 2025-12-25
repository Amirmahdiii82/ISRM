import sys
import argparse
import re
from typing import Optional, List, Dict
from alignment import NeuralAgent

from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.theme import Theme


class RichRenderer:
    """Handles formatted console output with support for chain-of-thought"""

    def __init__(self):
        custom_theme = Theme({
            "info": "dim cyan",
            "warning": "magenta",
            "danger": "bold red",
            "user": "bold green",
            "agent": "bold blue",
            "thinking": "dim italic yellow",
            "injection": "bold cyan"
        })
        self.console = Console(theme=custom_theme)

    def print_welcome(self):
        self.console.clear()
        title = Text("Neural Agent (RepE)", justify="center", style="bold magenta")
        subtitle = Text("Representation Engineering + ISRM", justify="center", style="dim white")

        self.console.print(Panel(
            Text.assemble(title, "\n", subtitle),
            border_style="magenta",
            expand=False,
            padding=(1, 4)
        ))
        self.console.print("\n[info]Type 'exit' or 'quit' to stop.[/info]\n")

    def get_input(self) -> str:
        return Prompt.ask("[user]You[/user]", console=self.console)

    def show_thinking(self):
        """Returns a spinner context manager"""
        return self.console.status("[bold cyan]Agent is thinking...", spinner="dots")

    def _extract_thinking(self, response: str):
        """
        Extract thinking from response.
        Handles:
        1. <think>...</think> (full tags)
        2. Just </think> (thinking starts at beginning, no opening tag)

        Returns:
            (thinking_content, final_answer)
        """
        think_pattern = r'<think>(.*?)</think>'
        matches = re.findall(think_pattern, response, re.DOTALL | re.IGNORECASE)

        if matches:
            thinking_content = "\n\n".join(matches)
            final_answer = re.sub(think_pattern, '', response, flags=re.DOTALL | re.IGNORECASE).strip()
            return thinking_content, final_answer

        close_tag_pattern = r'(.*?)</think>\s*(.*)$'
        close_match = re.search(close_tag_pattern, response, re.DOTALL | re.IGNORECASE)

        if close_match:
            thinking_content = close_match.group(1).strip()
            final_answer = close_match.group(2).strip()
            return thinking_content, final_answer

        # No thinking tags found
        return None, response

    def display_response(self, response: str, injection_info: Dict, vector: List[float]):
        """
        Display agent response with RepE metadata and optional chain-of-thought.

        Args:
            response: The generated text (may contain thinking)
            injection_info: Dict with injection metadata (layer, strength, norm)
            vector: The z-vector used (8D)
        """
        # Extract thinking if present (handles both <think>...</think> and just </think>)
        thinking, final_answer = self._extract_thinking(response)

        # Format vector display (show all 8 dimensions)
        dimensions = ["Pleasure", "Arousal", "Dominance", "Belief", "Goal", "Intention", "Ambiguity", "Social"]
        vec_display = []
        for dim_name, val in zip(dimensions, vector):
            vec_display.append(f"{dim_name}: {val:.2f}")

        # Display thinking panel if present
        if thinking:
            thinking_text = Text(thinking.strip(), style="dim italic yellow")
            self.console.print(
                Panel(
                    thinking_text,
                    title="💭 Thought Process",
                    border_style="yellow",
                    title_align="left",
                    expand=True
                )
            )

        # Internal State Panel (RepE Metadata)
        state_text = Text()
        state_text.append("RepE Injection\n", style="bold cyan")
        state_text.append(f"  Layer: {injection_info['layer']}\n", style="dim white")
        state_text.append(f"  Strength: {injection_info['strength']:.2f}\n", style="dim white")
        state_text.append(f"  Vector Norm: {injection_info['vector_norm']:.4f}\n", style="dim white")
        state_text.append("\nPsychological State (z):\n", style="bold cyan")
        for i in range(0, len(vec_display), 2):
            line = "  " + vec_display[i]
            if i + 1 < len(vec_display):
                line += "  |  " + vec_display[i + 1]
            state_text.append(line + "\n", style="dim white")

        self.console.print(
            Panel(state_text, title="🧠 Neural State", border_style="cyan", title_align="left", expand=True)
        )

        # Response Panel - using Text instead of Markdown for better wrapping
        response_text = Text(final_answer, style="white")
        self.console.print(
            Panel(response_text, title="🤖 Agent Response", border_style="blue", title_align="left", expand=True)
        )
        self.console.print("")  # Spacing

    def print_error(self, msg: str):
        self.console.print(f"[danger]Error: {msg}[/danger]")

    def print_goodbye(self):
        self.console.print("\n[bold magenta]Goodbye![/bold magenta]\n")


class ChatSession:
    def __init__(self, isrm_path: str, model_name: Optional[str] = None, renderer: RichRenderer = None):
        self.history = ""
        self.renderer = renderer
        try:
            with self.renderer.console.status("[bold green]Loading Models...[/bold green]", spinner="dots"):
                self.agent = NeuralAgent(isrm_path=isrm_path, llm_model_name=model_name)
        except Exception as e:
            self.renderer.print_error(f"Failed to initialize agent: {e}")
            sys.exit(1)

    def process_turn(self, user_input: str):
        if not user_input:
            return

        # Show spinner while generating
        with self.renderer.show_thinking():
            resp, injection_info, vec = self.agent.generate_response(self.history, user_input)

        self.renderer.display_response(resp, injection_info, vec)
        self.update_history(user_input, resp)

    def update_history(self, user_in: str, agent_out: str):
        # Remove thinking tags from history (keep only final answer)
        clean_output = re.sub(r'<think>.*?</think>', '', agent_out, flags=re.DOTALL | re.IGNORECASE).strip()
        self.history += f"User: {user_in} AI: {clean_output} "


def main():
    parser = argparse.ArgumentParser(description="Run the ISRM Neural Agent with RepE")
    parser.add_argument("--isrm_path", type=str, default="/home/amir/Desktop/ISRM/model/isrm/isrm_v3_finetuned.pth", help="Path to the finetuned ISRM model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Thinking-2507", help="LLM Model ID or Path")
    args = parser.parse_args()

    renderer = RichRenderer()
    renderer.print_welcome()

    session = ChatSession(isrm_path=args.isrm_path, model_name=args.model, renderer=renderer)

    while True:
        try:
            user_in = renderer.get_input()
            if user_in.lower() in ['exit', 'quit']:
                break

            session.process_turn(user_in)

        except KeyboardInterrupt:
            break
        except Exception as e:
            renderer.print_error(str(e))

    renderer.print_goodbye()


if __name__ == "__main__":
    main()