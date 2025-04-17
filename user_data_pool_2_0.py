from pymongo import MongoClient
from pymongo.errors import PyMongoError
import uuid
from datetime import datetime


try:
    mongo_client = MongoClient('mongodb://localhost:27017/',
                     maxPoolSize=50,
                     minPoolSize=1,
                     maxIdleTimeMS=30000,
                     waitQueueTimeoutMS=10000)
    mongo_db = mongo_client["Digi_chat_memory"]
    sessions_collection = mongo_db['sessions']
    messages_collection = mongo_db['messages']
    counters_collection = mongo_db["counters"]
    users_collection = mongo_db["user"]
    print("MongoDB connection established")
except Exception as e:
    print("Error connecting to MongoDB:", str(e))

def get_recent_session_id(user_id):
    try:
        # Query the sessions collection for the most recent session for the user
        recent_session = sessions_collection.find_one(
            {"user_id": user_id},  # Match the user_id
            sort=[("created_at", -1)]  # Sort by creation date in descending order
        )
        
        if recent_session:
            return recent_session["session_id"]
        else:
            return None  # If no session is found for the user
    except PyMongoError as e:
        return 'Error1.6 - ' + str(e)

# Function to get the next sequence value for user_id or session_id
def get_next_sequence_value(sequence_name):
    try:
        sequence_document = counters_collection.find_one_and_update(
            {"_id": sequence_name},
            {"$inc": {"sequence_value": 1}},
            return_document=True,
            upsert=True
        )
        return sequence_document["sequence_value"]
    except PyMongoError as e:
        return 'Error1.1 - ' + str(e)

# Function to get or create a user ID based on the user name
def get_or_create_user_id(user_name):

    try:
        # Check if user_name exists
        record = users_collection.find_one({"user_name": user_name})
        if record:
            # If user exists, return the user_id
            return record['user_id']
        else:
            # If user does not exist, insert new user and return new user_id
            new_user_id = get_next_sequence_value("user_id")
            users_collection.insert_one({"user_id": new_user_id, "user_name": user_name})
            return new_user_id
    except PyMongoError as e:
        return 'Error1.1 - ' + str(e)

# Function to create a new session and associate it with a user
def create_session(user_id):
    try:
        session_id = str(uuid.uuid4())  # Generate a unique session ID
        sessions_collection.insert_one({
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now()
        })
        return session_id
    except PyMongoError as e:
        return 'Error1.3 - ' + str(e)

# Function to store a message in a session
def save_message(session_id,user_id, role, message):
    try:
        message = {
            "session_id": session_id,  # Unique session ID
            "user_id": user_id,        # User who sent the message
            "role": role,              # 'human' or 'ai'
            "content": message,        # The message text
            "timestamp": datetime.utcnow()   # Store time in UTC format
        }
        messages_collection.insert_one(message)
    except PyMongoError as e:
        return 'Error1.4 - ' + str(e)
    
def load_session_messages(session_id):
    try:
        messages = messages_collection.find({"session_id": session_id}).sort("timestamp")
        return [{"role": msg["role"], "message": msg["message"]} for msg in messages]
    except PyMongoError as e:
        return 'Error1.5 - ' + str(e)
    
def get_first_message(session_id):
    first_message = messages_collection.find_one(
        {"session_id": session_id, "role": "human"},  # Only get user messages
        sort=[("timestamp", 1)]  # Sort by timestamp (oldest first)
    )
    
    return first_message["messages"] if first_message else None
    
# Function to get all sessions of a user (optional)
def get_user_sessions(user_id):
    try:
        sessions = sessions_collection.find({"user_id": user_id})
        return [{"session_id": session["session_id"], "created_at": session["created_at"]} for session in sessions]
    except PyMongoError as e:
        return 'Error1.6 - ' + str(e)
# Example usage
user_name = "Gogula"
user_id = get_or_create_user_id(user_name)

