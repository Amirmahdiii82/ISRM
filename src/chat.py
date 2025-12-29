import sys
import argparse
import re
from typing import Optional, Dict
from alignment import NeuralAgent, DEFAULT_BDI

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.theme import Theme

# BDI Persona Presets
BDI_PRESETS = {
    "neutral": DEFAULT_BDI.copy(),
    "skeptical": {"belief": 0.9, "goal": 0.6, "intention": 0.7, "ambiguity": 0.3, "social": 0.5},
    "trusting": {"belief": 0.1, "goal": 0.5, "intention": 0.4, "ambiguity": 0.6, "social": 0.8},
    "focused": {"belief": 0.5, "goal": 0.9, "intention": 0.8, "ambiguity": 0.2, "social": 0.6},
    "casual": {"belief": 0.4, "goal": 0.3, "intention": 0.2, "ambiguity": 0.7, "social": 0.9},
    "analytical": {"belief": 0.7, "goal": 0.7, "intention": 0.9, "ambiguity": 0.2, "social": 0.5},
    "friendly": {"belief": 0.3, "goal": 0.5, "intention": 0.5, "ambiguity": 0.5, "social": 1.0},
}

# Persona Descriptions for User Selection
PERSONA_DESCRIPTIONS = {
    "neutral": "Balanced response style (default)",
    "skeptical": "Questions claims, demands evidence, scientific mindset",
    "trusting": "Accepts information readily, open-minded approach",
    "focused": "Task-oriented, goal-driven, minimal distraction",
    "casual": "Relaxed, conversational, less structured",
    "analytical": "Deep reasoning, systematic thinking, logical",
    "friendly": "Warm, social, empathetic responses"
}


def get_persona_choice(console: Console) -> str:
    """Interactive persona selector with descriptions"""
    console.print("\n[bold cyan]Choose Agent Persona:[/bold cyan]\n")

    persona_list = list(PERSONA_DESCRIPTIONS.items())
    for i, (name, desc) in enumerate(persona_list, 1):
        console.print(f"  {i}. [bold]{name.capitalize()}[/bold]: {desc}")

    while True:
        choice = Prompt.ask(
            "\nSelect persona (number or name)",
            choices=[str(i) for i in range(1, len(persona_list) + 1)] + list(PERSONA_DESCRIPTIONS.keys()),
            default="1",
            console=console
        )

        if choice.isdigit():
            return persona_list[int(choice) - 1][0]
        return choice


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

    def print_welcome(self, persona: str = "neutral"):
        self.console.clear()
        title = Text("Hybrid Neural Agent", justify="center", style="bold magenta")
        subtitle = Text(f"Dynamic PAD + Static BDI (persona: {persona})", justify="center", style="dim white")

        self.console.print(Panel(
            Text.assemble(title, "\n", subtitle),
            border_style="magenta",
            expand=False,
            padding=(1, 4)
        ))
        self.console.print("[info]Commands: 'exit' or 'quit' to stop | '/persona <name>' to switch persona[/info]\n")

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

    def display_response(self, response: str, injection_info: Dict, state_info: Dict):
        """
        Display agent response with hybrid RepE metadata.

        Args:
            response: The generated text (may contain thinking)
            injection_info: Dict with injection metadata
            state_info: Dict with 'pad' (dynamic) and 'bdi' (static) values
        """
        thinking, final_answer = self._extract_thinking(response)

        # Display thinking panel if present
        if thinking:
            thinking_text = Text(thinking.strip(), style="dim italic yellow")
            self.console.print(
                Panel(thinking_text, title="Thought Process", border_style="yellow", title_align="left", expand=True)
            )

        # Build state display
        state_text = Text()

        # PAD (Dynamic)
        state_text.append("PAD [Dynamic]\n", style="bold green")
        pad = state_info["pad"]
        state_text.append(f"  Pleasure: {pad['pleasure']:.2f}  Arousal: {pad['arousal']:.2f}  Dominance: {pad['dominance']:.2f}\n", style="dim white")

        # BDI (Static)
        state_text.append("\nBDI [Static]\n", style="bold yellow")
        bdi = state_info["bdi"]
        state_text.append(f"  Belief: {bdi['belief']:.2f}  Goal: {bdi['goal']:.2f}  Intention: {bdi['intention']:.2f}\n", style="dim white")
        state_text.append(f"  Ambiguity: {bdi['ambiguity']:.2f}  Social: {bdi['social']:.2f}\n", style="dim white")

        # Injection info
        state_text.append("\nDual-Layer Injection\n", style="bold cyan")
        state_text.append(f"  PAD Layer: {injection_info['pad_layer']}  Strength: {injection_info['injection_strength']:.2f}\n", style="dim white")
        state_text.append(f"  BDI Layer: {injection_info['bdi_layer']}  Strength: {injection_info['bdi_strength']:.2f}\n", style="dim white")
        state_text.append(f"  v_pad_norm: {injection_info['v_pad_norm']:.3f}  v_bdi_norm: {injection_info['v_bdi_norm']:.3f}\n", style="dim white")

        self.console.print(
            Panel(state_text, title="Neural State", border_style="cyan", title_align="left", expand=True)
        )

        # Response Panel
        response_text = Text(final_answer, style="white")
        self.console.print(
            Panel(response_text, title="Agent Response", border_style="blue", title_align="left", expand=True)
        )
        self.console.print("")

    def print_error(self, msg: str):
        self.console.print(f"[danger]Error: {msg}[/danger]")

    def print_goodbye(self):
        self.console.print("\n[bold magenta]Goodbye![/bold magenta]\n")


class ChatSession:
    def __init__(self, isrm_path: str, model_name: Optional[str] = None,
                 bdi_config: Dict = None, bdi_strength: float = 1.0,
                 renderer: RichRenderer = None, persona: str = "neutral"):
        self.history = ""
        self.renderer = renderer
        self.current_persona = persona
        self.bdi_strength = bdi_strength
        self.isrm_path = isrm_path
        self.model_name = model_name
        try:
            with self.renderer.console.status("[bold green]Loading Models...[/bold green]", spinner="dots"):
                self.agent = NeuralAgent(
                    isrm_path=isrm_path,
                    llm_model_name=model_name,
                    bdi_config=bdi_config,
                    bdi_strength=bdi_strength
                )
        except Exception as e:
            self.renderer.print_error(f"Failed to initialize agent: {e}")
            sys.exit(1)

    def process_turn(self, user_input: str):
        if not user_input:
            return

        with self.renderer.show_thinking():
            resp, injection_info, state_info = self.agent.generate_response(self.history, user_input)

        self.renderer.display_response(resp, injection_info, state_info)
        self.update_history(user_input, resp)

    def update_history(self, user_in: str, agent_out: str):
        clean_output = re.sub(r'<think>.*?</think>', '', agent_out, flags=re.DOTALL | re.IGNORECASE).strip()
        self.history += f"User: {user_in} AI: {clean_output} "

    def change_persona(self, new_persona: str):
        """Change agent persona mid-conversation"""
        if new_persona not in BDI_PRESETS:
            self.renderer.print_error(f"Unknown persona '{new_persona}'. Available: {', '.join(BDI_PRESETS.keys())}")
            return False

        self.current_persona = new_persona
        new_bdi_config = BDI_PRESETS[new_persona].copy()

        # Update agent's BDI configuration
        self.agent.bdi_config = new_bdi_config

        self.renderer.console.print(f"\n[bold green]Persona changed to: {new_persona}[/bold green]")
        self.renderer.console.print(f"[dim]{PERSONA_DESCRIPTIONS[new_persona]}[/dim]\n")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Neural Agent: Dynamic PAD + Static BDI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--isrm_path", type=str,
                        default="./model/isrm/pad_encoder.pth",
                        help="Path to PAD encoder weights")
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen3-4B-Thinking-2507",
                        help="LLM Model ID or Path")
    parser.add_argument("--bdi-strength", type=float, default=1.0,
                        help="BDI component scaling (beta)")
    parser.add_argument("--belief", type=float, help="Override belief [0-1]")
    parser.add_argument("--goal", type=float, help="Override goal [0-1]")
    parser.add_argument("--intention", type=float, help="Override intention [0-1]")
    parser.add_argument("--ambiguity", type=float, help="Override ambiguity [0-1]")
    parser.add_argument("--social", type=float, help="Override social [0-1]")

    args = parser.parse_args()

    renderer = RichRenderer()

    selected_persona = get_persona_choice(renderer.console)

    bdi_config = BDI_PRESETS[selected_persona].copy()
    for key in ["belief", "goal", "intention", "ambiguity", "social"]:
        val = getattr(args, key)
        if val is not None:
            if not (0.0 <= val <= 1.0):
                print(f"Error: --{key} must be in [0, 1]")
                sys.exit(1)
            bdi_config[key] = val

    renderer.print_welcome(persona=selected_persona)

    session = ChatSession(
        isrm_path=args.isrm_path,
        model_name=args.model,
        bdi_config=bdi_config,
        bdi_strength=args.bdi_strength,
        renderer=renderer,
        persona=selected_persona
    )

    while True:
        try:
            user_in = renderer.get_input()
            if user_in.lower() in ['exit', 'quit']:
                break

            if user_in.lower().startswith('/persona'):
                parts = user_in.split(maxsplit=1)
                if len(parts) == 2:
                    session.change_persona(parts[1].strip())
                else:
                    renderer.console.print(f"[yellow]Current persona: {session.current_persona}[/yellow]")
                    renderer.console.print(f"[dim]Use '/persona <name>' to switch. Available: {', '.join(BDI_PRESETS.keys())}[/dim]\n")
                continue

            session.process_turn(user_in)
        except KeyboardInterrupt:
            break
        except Exception as e:
            renderer.print_error(str(e))

    renderer.print_goodbye()


if __name__ == "__main__":
    main()