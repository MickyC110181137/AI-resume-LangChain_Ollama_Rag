from flask import Flask, request, jsonify
import json

# 讀取 JSON 檔案
def load_users():
    with open("users.json", "r", encoding="utf-8") as file:
        return json.load(file)

def save_users(users):
    with open("users.json", "w", encoding="utf-8") as file:
        json.dump(users, file, ensure_ascii=False, indent=4)

# 初次讀取
users = load_users()
print(users)

# Flask-------------------------------------------------------
app = Flask(__name__)

# 獲取所有用戶的信息
@app.route("/user", methods=["GET"])
def get_user():
    return jsonify({"users": users})

# 創建新的用戶
@app.route("/user", methods=["POST"])
def create_user():
    request_data = request.get_json()
    new_user = {"name": request_data["name"], "message": [], "llm_anwser": []}
    users.append(new_user)
    save_users(users)  # 保存到 JSON 檔案
    return jsonify(new_user), 201

# 在指定用戶中添加訊息
@app.route("/user/<string:name>/message", methods=["POST"])
def add_message_to_user(name):
    request_data = request.get_json()
    for user in users:
        if user["name"] == name:
            new_message = request_data["message"]
            user["message"].append(new_message)
            save_users(users)  # 保存到 JSON 檔案
            return jsonify({"message": f"Course {new_message} added to user {name}"}), 201
    return jsonify({"message": "usesr not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
# Flask-------------------------------------------------------

