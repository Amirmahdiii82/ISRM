import sys
import argparse
import time
from typing import Optional, List
from alignment import Agent_OpenEnded

from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.theme import Theme
from rich.spinner import Spinner

# --- UI / Display Logic (Rich TUI) ---
class RichRenderer:
    """Handles formatted console output using the Rich library."""
    
    def __init__(self):
        custom_theme = Theme({
            "info": "dim cyan",
            "warning": "magenta",
            "danger": "bold red",
            "user": "bold green",
            "agent": "bold blue",
            "system": "italic yellow"
        })
        self.console = Console(theme=custom_theme)

    def print_welcome(self):
        self.console.clear()
        title = Text("🤖 Qwen-ISRM Agent", justify="center", style="bold magenta")
        subtitle = Text("Interactive Latent Space Response Model", justify="center", style="dim white")
        
        self.console.print(Panel(
            Text.assemble(title, "\n", subtitle),
            border_style="magenta",
            expand=False,
            padding=(1, 4)
        ))
        self.console.print("\n[info]Type 'exit' or 'quit' to stop.[/info]\n")

    def get_input(self) -> str:
        # Use rich's Prompt for consistent styling
        return Prompt.ask("[user]You[/user]", console=self.console)

    def show_thinking(self):
        """Returns a spinner context manager."""
        return self.console.status("[bold cyan]Agent is thinking...", spinner="dots")

    def display_response(self, response: str, system_prompt: str, vector: List[float]):
        # Parse internals
        sys_lines = system_prompt.strip().splitlines()
        mode_line = sys_lines[0] if sys_lines else "MODE: UNKNOWN"
        instruction = sys_lines[-1] if len(sys_lines) > 1 else ""
        
        # Format vector
        vec_str = ", ".join([f"{v:.2f}" for v in vector[:4]])
        
        # Create layouts
        grid = Layout()
        grid.split_column(
            Layout(name="internals", size=4),
            Layout(name="response")
        )

        # Internal State Panel
        state_text = Text()
        state_text.append(f"{mode_line}\n", style="bold yellow")
        state_text.append(f"Instruction: ", style="dim yellow")
        state_text.append(f"{instruction}\n", style="system")
        state_text.append(f"Vector (z): [{vec_str}...]", style="dim white")

        grid["internals"].update(
            Panel(state_text, title="🧠 Internal State", border_style="yellow", title_align="left")
        )

        # Response Panel
        # Using Markdown for nice rendering of LLM output
        md_response = Markdown(response)
        grid["response"].update(
            Panel(md_response, title="🤖 Agent Response", border_style="blue", title_align="left")
        )

        self.console.print(grid)
        self.console.print("") # Spacing

    def print_error(self, msg: str):
        self.console.print(f"[danger]❌ Error: {msg}[/danger]")

    def print_goodbye(self):
        self.console.print("\n[bold magenta]👋 Goodbye![/bold magenta]\n")


# --- Chat Logic ---
class ChatSession:
    def __init__(self, isrm_path: str, model_name: Optional[str] = None, renderer: RichRenderer = None):
        self.history = ""
        self.renderer = renderer
        try:
            with self.renderer.console.status("[bold green]Loading Models...[/bold green]", spinner="dots"):
                self.agent = Agent_OpenEnded(isrm_path=isrm_path, llm_model_name=model_name)
        except Exception as e:
            self.renderer.print_error(f"Failed to initialize agent: {e}")
            sys.exit(1)

    def process_turn(self, user_input: str):
        if not user_input:
            return

        # Show spinner while generating
        with self.renderer.show_thinking():
            # Simulate a tiny delay for UI feel if local inference is too instant (optional) 
            # time.sleep(0.5) 
            resp, system_prompt, vec = self.agent.generate_response(self.history, user_input)
        
        self.renderer.display_response(resp, system_prompt, vec)
        self.update_history(user_input, resp)

    def update_history(self, user_in: str, agent_out: str):
        self.history += f"User: {user_in} AI: {agent_out} "


def main():
    parser = argparse.ArgumentParser(description="Run the ISRM Chat Client")
    parser.add_argument("--isrm_path", type=str, default="/home/amir/Desktop/ISRM/model/isrm/isrm_v3_finetuned.pth", help="Path to the finetuned ISRM model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="LLM Model ID or Path")
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
