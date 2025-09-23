from pathlib import Path
import json

class KnowledgeBase:
    """
    This class manages the creation and filtering of knowledge
    for injection into the agent templates
    """
    def __init__(self):
        print("Initialize the knowledge base")
        self.knowledge_base = ""
        self.knowledge_base_mappings = {}

    
    def create_broad_adverb_knowledge_base(self):
        """
        Knowledge base looks like this:
        KNOWLEDGE ABOUT ADVERB CATEGORIES
        A. CIRCUMSTANCE ADVERBS provide information about...
        B. STANCE ADVERBS provide information about ...

        Mappings dictionary is also created
        {
            "A": "CIRCUMSTANCE",
            "B": "STANCE",
            ...
        }
        """

        knowledge_base_path = Path(__file__).resolve().parent / "adverbs.json"
        knowledge_data = None
        # Get the knowledge data from the data source
        try:
            with knowledge_base_path.open("r", encoding="utf-8") as kb_file:
                knowledge_data = json.load(kb_file)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Knowledge base not found at {knowledge_base_path} at runtime.") from exc
        
        # Start building the knowledge base
        self.knowledge_base = "KNOWLEDGE ABOUT ADVERB CATEGORIES:\n\n"

        # Letter choices
        letter_choices = ["A", "B", "C", "D"]
        for idx, (category_name, category_info) in enumerate(knowledge_data["Adverbials"].items()):
            description = category_info["description"]
            title = f"{letter_choices[idx]}. {category_name.upper()}"
            if "adverbs" not in category_name.lower():
                title += " ADVERBS"
            self.knowledge_base += f"{title}: {description}\n"
            self.knowledge_base_mappings[letter_choices[idx]] = category_name.upper()

        print(f"Knowledge base prepared: {self.knowledge_base}")
    
    def get_knowledge_base(self):
        """
        Returns the string containing the created knowledge base.
        If the knowledge base has not been created yet, it will return ""
        """
        return self.knowledge_base
    
    def get_knowledge_base_mappings(self):
        """
        Returns a dictionary containing the mappings of answer keys to categories:
        e.g.,
        {
            "A": "CIRCUMSTANCE",
            "B": "STANCE"
            ...
        }
        """
        return self.knowledge_base_mappings