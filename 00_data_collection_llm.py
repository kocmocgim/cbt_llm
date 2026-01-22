import psycopg2
import requests
import json
import time

class OllamaAnalyzer:
    """
    Minimal analyzer class for interacting with the Ollama API.
    Reads data from PostgreSQL, processes it via Ollama, and saves the result.
    """

    def __init__(self):
        # Database configuration - REPLACE WITH YOUR CREDENTIALS
        self.db_config = {
            "host": "localhost",
            "user": "your_username",
            "password": "your_password",
            "port": 5432,
            "database": "your_database_name"
        }
        
        # Ollama API configuration
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "qwen2.5:14b"  # Specify your model here
        self.conn = None
        
        # Prompt template is initialized as None. MUST be loaded from file.
        self.prompt_template = None

    def load_prompt(self, file_path):
        """
        Strictly loads the prompt from a file. 
        Raises an exception if the file is missing or empty.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if not content:
                    raise ValueError("Prompt file is empty")
                self.prompt_template = content
                print(f"✅ Prompt loaded successfully from {file_path}")
        except Exception as e:
            # HARD ERROR: Re-raise exception to stop execution immediately
            raise RuntimeError(f"CRITICAL: Failed to load prompt from '{file_path}'. {e}")

    def connect_to_db(self):
        """Connects to the PostgreSQL database."""
        try:
            if self.conn and not self.conn.closed:
                self.conn.close()
                
            self.conn = psycopg2.connect(**self.db_config)
            self.conn.autocommit = True
            print("✅ DB: Connected successfully")
            return True
        except Exception as e:
            print(f"❌ DB: Connection failed: {e}")
            return False

    def load_batch_from_db(self, batch_size=10):
        """
        Loads a batch of unprocessed records from the database.
        Returns a list of tuples: (id, content, tag, input_id)
        """
        try:
            with self.conn.cursor() as cursor:
                # Select records where is_cbt = 0 (unprocessed)
                # FOR UPDATE SKIP LOCKED prevents race conditions if multiple scripts run
                cursor.execute("""
                    SELECT d.id, d.content, d.tag, t.input_id 
                    FROM _input_data d
                    JOIN input_text_attributes t ON d.id = t.input_id
                    WHERE t.is_cbt = 0
                    ORDER BY LENGTH(d.content) ASC
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                """, (batch_size,))

                entries = cursor.fetchall()

                # Mark records as 'in progress' (is_cbt = -1) to prevent double processing
                if entries:
                    input_ids = [entry[3] for entry in entries]
                    placeholders = ','.join(['%s'] * len(input_ids))
                    cursor.execute(f"""
                        UPDATE input_text_attributes 
                        SET is_cbt = -1 
                        WHERE input_id IN ({placeholders})
                    """, input_ids)

                print(f"📦 DB: Loaded {len(entries)} records")
                return entries
        except Exception as e:
            print(f"❌ DB: Error loading batch: {e}")
            # Signal that we might need a reconnect
            return None

    def process_content(self, content):
        """Sends content to the Ollama API and returns the generated text."""
        # Ensure prompt is loaded before processing
        if not self.prompt_template:
            raise RuntimeError("Prompt template is not loaded!")

        full_prompt = self.prompt_template.replace("{CONTENT}", content)

        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "seed": 42
            }
        }

        try:
            response = requests.post(self.ollama_url, headers=headers, data=json.dumps(data), timeout=180)

            if response.status_code == 200:
                try:
                    result = response.json().get("response", "")
                    return result
                except json.JSONDecodeError:
                    print("❌ Ollama: Invalid JSON response")
                    return None
            else:
                print(f"❌ Ollama: HTTP Error {response.status_code}")
                return None

        except Exception as e:
            print(f"❌ Ollama: API Request failed: {e}")
            return None

    def save_result(self, input_id, result):
        """Saves the LLM result back to the database."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE input_text_attributes 
                    SET is_cbt = 1, cbt_result = %s, folder_title = %s 
                    WHERE input_id = %s
                """, (result, self.model, input_id))
                
            print(f"💾 DB: Result saved for ID {input_id}")
            return True
        except Exception as e:
            print(f"❌ DB: Error saving result: {e}")
            return False

    def run(self):
        """Main execution loop."""
        print("🚀 Starting Minimal Ollama Analyzer")

        # STRICT: Attempt to load prompt first. 
        self.load_prompt("LLM_prompt.txt")

        # Initial connection
        if not self.connect_to_db():
            print("❌ Initial DB connection failed. Exiting.")
            return

        while True:
            try:
                # Load a small batch
                batch = self.load_batch_from_db(batch_size=10)

                # Check if DB error occurred (batch is None) vs just empty (batch is [])
                if batch is None:
                    print("⚠️ DB Connection issue detected. Attempting reconnect...")
                    time.sleep(5)
                    self.connect_to_db()
                    continue

                if not batch:
                    print("💤 No pending records. Sleeping for 30 seconds...")
                    time.sleep(30)
                    continue

                for entry in batch:
                    entry_id, content, tag, input_id = entry
                    
                    if not content or not content.strip():
                        print(f"⚠️ Skipping empty content for ID {entry_id}")
                        self.save_result(input_id, "Empty Content")
                        continue

                    print(f"🔄 Processing ID: {entry_id}...")
                    result = self.process_content(content)

                    if result:
                        self.save_result(input_id, result)
                    else:
                        print(f"❌ Failed to process ID {entry_id}")
                    
                    # Small delay to be gentle on the API
                    time.sleep(1)

            except KeyboardInterrupt:
                print("\n⛔ Stopped by user")
                break
            except Exception as e:
                print(f"❌ Unexpected error in loop: {e}")
                time.sleep(10)

        if self.conn:
            self.conn.close()

if __name__ == "__main__":
    analyzer = OllamaAnalyzer()
    analyzer.run()