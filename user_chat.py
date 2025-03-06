"""
This script provides an interactive chat interface with user registration,
chat history saving, and emotional state tracking using OpenAI or HF models.
"""

import os
import sqlite3
from datetime import datetime
import json
import re
import hashlib
import getpass
import torch
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from openai import  OpenAI
from typing import List, Dict, Optional, Tuple
from fire import Fire
from prompting.llama_prompt import modified_extes_support_strategies


@dataclass
class ModelConfig:
    model_type: str  # "hf" or "openai"
    model_path: str
    use_4bit: bool = False
    temperature: float = 0.8
    top_p: float = 0.9
    max_tokens: int = 1024

@dataclass
class UserProfile:
    user_id: str
    age: Optional[int] = None
    name: Optional[str] = None
    gender: Optional[str] = None
    occupation: Optional[str] = None
    zipcode: Optional[str] = None
    emotional_state: Optional[str] = None
    last_interaction: Optional[str] = None

class EnhancedDatabaseManager():
    def __init__(self, db_path: str = "user_data.db"):
        self.db_path = db_path
        self._init_db()
        self._init_chat_history_table()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    name TEXT,
                    age INTEGER,
                    gender TEXT,
                    zipcode TEXT,
                    occupation TEXT,
                    emotional_state TEXT,
                    last_interaction TEXT
                )
            """)
    
    def _init_chat_history_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    timestamp TEXT,
                    role TEXT,
                    content TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
    
    def save_message(self, user_id: str, role: str, content: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO chat_history 
                (user_id, timestamp, role, content)
                VALUES (?, ?, ?, ?)
            """, (
                user_id,
                datetime.now().isoformat(),
                role,
                content
            ))
            
    def get_chat_history(self, user_id: str, limit: int = 50) -> List[Dict[str, str]]:
        with sqlite3.connect(self.db_path) as conn:
            results = conn.execute("""
                SELECT role, content FROM chat_history 
                WHERE user_id = ? 
                ORDER BY timestamp ASC 
                LIMIT ?
            """, (user_id, limit)).fetchall()
            
            return [{"role": row[0], "content": row[1]} for row in results]
    
    def update_emotional_state(self, user_id: str, emotional_state: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE users 
                SET emotional_state = ?, last_interaction = ?
                WHERE user_id = ?
            """, (
                emotional_state,
                datetime.now().isoformat(),
                user_id
            ))
    
    def username_exists(self, username: str) -> bool:
        """Check if a username already exists in the database."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM users WHERE username = ?", (username,)
            ).fetchone()
            return result[0] > 0
    
    def hash_password(self, password: str) -> str:
        """Hash a password for storing."""
        return hashlib.sha256(password.encode('utf-8')).hexdigest()
    
    def verify_credentials(self, username: str, password: str) -> Optional[str]:
        """Verify username and password, returns user_id if valid."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT user_id, password_hash FROM users WHERE username = ?", 
                (username,)
            ).fetchone()
            
            if result and result[1] == self.hash_password(password):
                return result[0]
            return None
    
    def get_user_by_username(self, username: str) -> Optional[UserProfile]:
        """Get a user by username."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT user_id, name, age, gender, occupation, zipcode, emotional_state, last_interaction " +
                "FROM users WHERE username = ?", 
                (username,)
            ).fetchone()
            
            if result:
                return UserProfile(
                    user_id=result[0],
                    name=result[1],
                    age=result[2],
                    gender=result[3],
                    occupation=result[4],
                    zipcode=result[5],
                    emotional_state=result[6],
                    last_interaction=result[7]
                )
            return None

def authenticate_user(db_manager: EnhancedDatabaseManager) -> Tuple[UserProfile, bool]:
    """Handle user authentication (login or signup) and return user profile and is_new flag."""
    print("\n===== Anchor Emotional Support Chat =====\n")
    print("1. Login")
    print("2. Sign up")
    
    while True:
        choice = input("\nPlease select an option (1/2): ").strip()
        
        if choice == "1":
            # Login process
            print("\n----- Login -----")
            username = input("Username: ").strip()
            password = getpass.getpass("Password: ")
            
            user_id = db_manager.verify_credentials(username, password)
            if user_id:
                user_profile = db_manager.get_user_by_username(username)
                print(f"\n✓ Welcome back, {user_profile.name}!")
                return user_profile, False
            else:
                print("\n✗ Invalid username or password.")
                continue
                
        elif choice == "2":
            # Sign up process
            print("\n----- Sign up -----")
            username = input("Choose a username: ").strip()
            
            # Validate username
            if not username or len(username) < 3:
                print("Username must be at least 3 characters long.")
                continue
                
            # Check if username exists
            if db_manager.username_exists(username):
                print("Username already taken. Please choose a different one.")
                continue
                
            # Get password
            while True:
                password = getpass.getpass("Choose a password (min 6 characters): ")
                if len(password) < 6:
                    print("Password must be at least 6 characters long.")
                    continue
                    
                confirm_password = getpass.getpass("Confirm password: ")
                if password != confirm_password:
                    print("Passwords don't match. Please try again.")
                    continue
                break
                
            # Get user info for new account
            user_info = get_user_info()
            
            # Create user ID
            user_id = f"{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create and save user
            user_profile = UserProfile(
                user_id=user_id,
                name=user_info['name'],
                age=user_info['age'],
                gender=user_info['gender'],
                zipcode=user_info.get('zipcode'),
                emotional_state="unknown",
                last_interaction=datetime.now().isoformat()
            )
            
            # Save user with password
            with sqlite3.connect(db_manager.db_path) as conn:
                conn.execute("""
                    INSERT INTO users 
                    (user_id, username, password_hash, name, age, gender, zipcode, emotional_state, last_interaction)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_profile.user_id,
                    username,
                    db_manager.hash_password(password),
                    user_profile.name,
                    user_profile.age,
                    user_profile.gender,
                    user_profile.zipcode,
                    user_profile.emotional_state,
                    user_profile.last_interaction
                ))
            
            print(f"\n✓ Account created successfully! Welcome, {user_profile.name}!")
            return user_profile, True
            
        else:
            print("Invalid option. Please choose 1 or 2.")

def get_user_info() -> Dict[str, any]:
    """Prompt the user for their personal information."""
    print("\nPlease provide some information about yourself:")
    
    name = input("Name: ").strip()
    
    # Validate age
    age = None
    while age is None:
        try:
            age_input = input("Age: ").strip()
            if age_input:
                age = int(age_input)
                if age <= 0 or age > 120:
                    print("Please enter a valid age between 1 and 120.")
                    age = None
        except ValueError:
            print("Please enter a valid number for age.")
    
    gender = input("Gender: ").strip()
    
    # Validate zip code
    zipcode = None
    while zipcode is None:
        zipcode_input = input("Zip code: ").strip()
        if re.match(r'^\d{5}(-\d{4})?$', zipcode_input) or not zipcode_input:
            zipcode = zipcode_input
        else:
            print("Please enter a valid 5-digit zip code or 5+4 digit zip code (e.g., 12345 or 12345-6789).")
    
    return {
        "name": name,
        "age": age,
        "gender": gender,
        "zipcode": zipcode
    }

class EnhancedChatModel():
    def __init__(self, config: ModelConfig):
        self.config = config
        self.db_manager = EnhancedDatabaseManager()
        if config.model_type == "hf":
            self._setup_hf_model()
        elif config.model_type == "openai":
            self._setup_openai()
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
        self.emotion_check_interval = 5  # Check emotions every 5 messages

    def _setup_hf_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.config.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_storage=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.bfloat16
            ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def _setup_openai(self):
        self.client = OpenAI()

    def get_strategy(self, messages: List[Dict[str, str]]) -> Optional[str]:
        strategies = list(modified_extes_support_strategies.keys())
        strategy_descriptions = "\n".join([f"{i+1}. {s}: {modified_extes_support_strategies[s]}" 
                                         for i, s in enumerate(strategies)])
        
        strategy_prompt = f"""Based on the user's message, choose the most appropriate emotional support strategy from the following options:
{strategy_descriptions}
{len(strategies)+1}. No strategy: Use general emotional support without a specific strategy

Respond with ONLY the number of your chosen strategy."""

        if self.config.model_type == "hf":
            return self._get_strategy_hf(messages, strategy_prompt, strategies)
        else:
            return self._get_strategy_openai(messages, strategy_prompt, strategies)

    def _get_strategy_hf(self, messages, strategy_prompt, strategies):
        prompt = [[
            {"role": "system", "content": "You are an expert at selecting appropriate emotional support strategies."},
            *messages,
            {"role": "user", "content": strategy_prompt}
        ]]
        
        input_prompt = self.tokenizer.apply_chat_template(
            prompt[0],
            add_generation_prompt=True,
            tokenize=False,
        )
        
        inputs = self.tokenizer(input_prompt, return_tensors='pt', padding=True).to(self.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        response = self.tokenizer.batch_decode(
            output[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]
        
        try:
            strategy_id = ''.join(filter(str.isdigit, response.strip()))
            return strategies[int(strategy_id) - 1] if strategy_id and int(strategy_id) <= len(strategies) else None
        except:
            return None

    def _get_strategy_openai(self, messages, strategy_prompt, strategies):
        formatted_messages = [
            {"role": "system", "content": "You are an expert at selecting appropriate emotional support strategies."},
            *messages,
            {"role": "user", "content": strategy_prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.config.model_path,
            messages=formatted_messages,
            temperature=0,
            max_tokens=10
        )
        
        try:
            strategy_id = ''.join(filter(str.isdigit, response.choices[0].message.content.strip()))
            return strategies[int(strategy_id) - 1] if strategy_id and int(strategy_id) <= len(strategies) else None
        except:
            return None

    def generate_response(self, sys_msg: str, messages: List[Dict[str, str]]) -> str:
        if self.config.model_type == "hf":
            return self._generate_response_hf(sys_msg, messages)
        else:
            return self._generate_response_openai(sys_msg, messages)

    def _generate_response_hf(self, sys_msg: str, messages: List[Dict[str, str]]) -> str:
        prompt = [[{"role": "system", "content": sys_msg}, *messages]]
        input_prompt = self.tokenizer.apply_chat_template(
            prompt[0],
            add_generation_prompt=True,
            tokenize=False,
        )
        
        inputs = self.tokenizer(input_prompt, return_tensors='pt', padding=True).to(self.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_tokens,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        return self.tokenizer.batch_decode(
            output[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]

    def _generate_response_openai(self, sys_msg: str, messages: List[Dict[str, str]]) -> str:
        formatted_messages = [{"role": "system", "content": sys_msg}, *messages]
        response = self.client.chat.completions.create(
            model=self.config.model_path,
            messages=formatted_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p
        )
        return response.choices[0].message.content

    def _extract_user_info(self, messages: List[Dict[str, str]]) -> Dict[str, any]:
        analysis_prompt = """Based on the conversation history, please analyze and extract the following information about the user. 
        Respond in JSON format with these fields (use null if uncertain):
        {
            "age": number or null,
            "gender": string or null,
            "occupation": string or null,
            "zipcode": string or null,
            "emotional_state": string or null
        }
        Only include information that has been directly stated or can be confidently inferred."""

        if self.config.model_type == "openai":
            response = self.client.chat.completions.create(
                model=self.config.model_path,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing conversations and extracting user information."},
                    *messages,
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0
            )
            try:
                return json.loads(response.choices[0].message.content)
            except:
                return {}
        else:
            # Similar implementation for HuggingFace models
            # ... (implement HF version of information extraction)
            return {}

    def introduce_self(self) -> str:
        intro_prompt = """Introduce yourself as Anchor, an AI emotional support companion. 
        Be warm and welcoming, and ask the user how they've been feeling lately. 
        Keep it concise but friendly."""
        
        if self.config.model_type == "openai":
            response = self.client.chat.completions.create(
                model=self.config.model_path,
                messages=[
                    {"role": "system", "content": "You are a warm and empathetic AI companion."},
                    {"role": "user", "content": intro_prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        else:
            # Implement HF version
            return self._generate_response_hf("You are a warm and empathetic AI companion.", 
                                           [{"role": "user", "content": intro_prompt}])
    
    def analyze_emotional_state(self, messages: List[Dict[str, str]]) -> str:
        """Analyze the user's emotional state based on recent messages."""
        emotion_prompt = """Based on the user's recent messages, what is their current emotional state?
        Provide a brief assessment (1-3 words) of their primary emotional state such as 'happy', 
        'anxious', 'stressed', 'calm', 'angry', 'confused', etc. 
        If uncertain, respond with 'unknown'."""

        if self.config.model_type == "openai":
            response = self.client.chat.completions.create(
                model=self.config.model_path,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing emotional states."},
                    *messages[-5:],  # Use the 5 most recent messages for analysis
                    {"role": "user", "content": emotion_prompt}
                ],
                temperature=0,
                max_tokens=50
            )
            return response.choices[0].message.content.strip().lower()
        else:
            # For HuggingFace models
            prompt = [[
                {"role": "system", "content": "You are an expert at analyzing emotional states."},
                *messages[-5:],
                {"role": "user", "content": emotion_prompt}
            ]]
            
            input_prompt = self.tokenizer.apply_chat_template(
                prompt[0],
                add_generation_prompt=True,
                tokenize=False,
            )
            
            inputs = self.tokenizer(input_prompt, return_tensors='pt', padding=True).to(self.device)
            output = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
            response = self.tokenizer.batch_decode(
                output[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )[0]
            
            return response.strip().lower()

def get_sys_msg_with_strategy(strategy: Optional[str]) -> str:
    if strategy is None:
        return "You are a helpful, precise and accurate emotional support expert"

    description = modified_extes_support_strategies[strategy]
    return f"""You are a helpful and caring AI which is an expert in emotional support.
            A user has come to you with emotional challenges, distress or anxiety.
            Use "{strategy}" strategy ({description}) for answering the user.
            Be as inquisitive, compassionate, and helpful as possible, while keeping your answers
            concise."""

def enhanced_chat_loop(model_config: ModelConfig):
    """Main chat loop with user registration/login and emotion tracking."""
    chat_model = EnhancedChatModel(model_config)
    
    # Authenticate user (login or signup)
    user_profile, is_new_user = authenticate_user(chat_model.db_manager)
    
    # Initialize chat
    messages = []
    strategy = None
    message_counter = 0
    
    # Load previous chat history if existing user
    if not is_new_user:
        previous_messages = chat_model.db_manager.get_chat_history(user_profile.user_id, limit=10)
        if previous_messages:
            print("\nLoading your previous conversation...")
            
            # Only add the last few messages to the current session (to avoid context window issues)
            if len(previous_messages) > 5:
                messages = previous_messages[-5:]  # Get last 5 messages
                print(f"Loaded the last {len(messages)} messages from your previous conversation.")
            else:
                messages = previous_messages
                print(f"Loaded {len(messages)} messages from your previous conversation.")
            messages.append({"role": "system", "content": "This was your previous conversation with the user. Now you are starting a new session with the user."})
    
    # Generate and display introduction
    if is_new_user or not messages:
        intro_message = chat_model.introduce_self()
        print(f'\033[34mAnchor\033[0m: {intro_message}')
        messages.append({"role": "assistant", "content": intro_message})
        chat_model.db_manager.save_message(user_profile.user_id, "assistant", intro_message)
    
    print(f"\nWelcome, \033[33m{user_profile.name}\033[0m! Chat initialized with {model_config.model_type} model.")
    print("Type 'reset' to start a new conversation or 'quit' to exit\n")
    
    while True:
        user_msg = input("\033[33mYou\033[0m: ").strip()
        
        if user_msg.lower() == 'quit':
            break
        elif user_msg.lower() == 'reset':
            messages = []
            strategy = None
            message_counter = 0
            print("Starting new conversation...")
            continue
            
        messages.append({"role": "user", "content": user_msg})
        chat_model.db_manager.save_message(user_profile.user_id, "user", user_msg)
        
        message_counter += 1
        
        # Periodically check emotional state
        if message_counter % chat_model.emotion_check_interval == 0:
            emotional_state = chat_model.analyze_emotional_state(messages)
            chat_model.db_manager.update_emotional_state(user_profile.user_id, emotional_state)
            
            # Save as a system message in chat history for tracking
            chat_model.db_manager.save_message(
                user_profile.user_id, 
                "system", 
                f"Detected emotional state: {emotional_state}"
            )
            
            print(f"\n[\033[35mSystem\033[0m: Detected emotional state: {emotional_state}]\n")
        
        # Select strategy and generate response
        strategy = chat_model.get_strategy(messages)
        sys_msg = get_sys_msg_with_strategy(strategy)
        
        assistant_msg = chat_model.generate_response(sys_msg, messages)
        print(f'\033[34mAnchor\033[0m \033[22;38;5;8m[{strategy}]\033[0m: {assistant_msg}')
        
        messages.append({"role": "assistant", "content": assistant_msg})
        chat_model.db_manager.save_message(user_profile.user_id, "assistant", assistant_msg)

def main(
    model_type: str = "openai",
    model_path: str = "gpt-4o",
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_tokens: int = 1024
):
    """
    Start an interactive chat session with the specified model, with user authentication
    and emotional state tracking.
    
    Args:
        model_type: Either "hf" for HuggingFace models or "openai" for OpenAI models
        model_path: Model identifier (HF model path or OpenAI model name)
        temperature: Temperature for response generation
        top_p: Top-p sampling parameter
        max_tokens: Maximum number of tokens in response
    """
    config = ModelConfig(
        model_type=model_type,
        model_path=model_path,
        use_4bit=False,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    
    enhanced_chat_loop(config)

if __name__ == '__main__':
    Fire(main)
