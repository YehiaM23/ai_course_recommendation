
# Smart Course Recommender for Diverse IT Students
# Modified version of the AI Curriculum Planner project

import random
import json
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

# Part 1: Curriculum Graph
courses = [
    "IntroProgramming", "MathBasics", "WebDevelopment", "OOP",
    "DatabaseSystems", "Networks", "CloudComputing", "CyberSecurity",
    "AI_Fundamentals", "DeepLearning", "DataVisualization"
]

prerequisites = {
    "OOP": ["IntroProgramming"],
    "WebDevelopment": ["IntroProgramming"],
    "DatabaseSystems": ["MathBasics"],
    "Networks": ["DatabaseSystems"],
    "CloudComputing": ["Networks"],
    "CyberSecurity": ["Networks"],
    "AI_Fundamentals": ["OOP", "MathBasics"],
    "DeepLearning": ["AI_Fundamentals"],
    "DataVisualization": ["DatabaseSystems"]
}

interest_areas = ["Cloud", "Cyber", "AI", "Frontend", "Visualization"]
grade_scale = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}

G = nx.DiGraph()
G.add_nodes_from(courses)
for course, pres in prerequisites.items():
    for p in pres:
        G.add_edge(p, course)

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=2200, node_color='lightgreen', font_size=9, arrows=True)
plt.title("Updated Curriculum Graph")
plt.savefig("updated_curriculum_graph.png")
plt.close()

# Part 2: Student Generation
def generate_student(student_id):
    completed = []
    grades = {}
    available = courses.copy()
    random.shuffle(available)
    for course in available:
        if all(p in completed for p in prerequisites.get(course, [])):
            completed.append(course)
            grades[course] = random.choice(list(grade_scale.keys()))
            if len(completed) >= random.randint(3, 6):
                break
    gpa = round(sum(grade_scale[g] for g in grades.values()) / len(grades), 2)
    interests = random.sample(interest_areas, k=random.randint(1, 2))
    return {
        "id": student_id,
        "completed_courses": completed,
        "grades": grades,
        "gpa": gpa,
        "term": random.randint(1, 8),
        "interests": interests
    }

students = [generate_student(i) for i in range(1, 101)]
with open("sample_students.json", "w") as f:
    json.dump(students, f, indent=2)

# Part 3: RL Recommendation
def is_course_available(course, completed):
    return all(p in completed for p in prerequisites.get(course, []))

def compute_reward(student, selected):
    reward = 0
    for course in selected:
        if not is_course_available(course, student["completed_courses"]):
            reward -= 4
        elif course in student["completed_courses"]:
            reward -= 2
        else:
            if any(interest.lower() in course.lower() for interest in student["interests"]):
                reward += 3
            if any(course in prerequisites.get(c, []) for c in courses):
                reward += 2
    if student["gpa"] > 3.0:
        reward += 1
    return reward

Q_table = defaultdict(float)
alpha = 0.5
gamma = 0.9

for ep in range(10):
    student = random.choice(students)
    state = (tuple(sorted(student["completed_courses"])), round(student["gpa"], 1), student["term"], tuple(sorted(student["interests"])))
    possible_courses = [c for c in courses if is_course_available(c, student["completed_courses"]) and c not in student["completed_courses"]]
    actions = [tuple(random.sample(possible_courses, k=min(3, len(possible_courses)))) for _ in range(5)]
    action = random.choice(actions)
    reward = compute_reward(student, action)
    Q_table[(state, action)] += alpha * (reward + gamma * 0 - Q_table[(state, action)])

with open("q_table.json", "w") as f:
    json.dump({str(k): v for k, v in Q_table.items()}, f, indent=2)

print("✅ Modified Project Completed")
