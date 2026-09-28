# career_matcher.py
# single model: clusters students and jobs, then maps one new student to job roles.
# steps: load data -> clean -> cluster jobs into tiers -> cluster students into levels
#        -> take terminal input -> find student level -> suggest roles -> upgrade path.
# needs: pip install pandas numpy scikit-learn

import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# section 1: settings
# csv files are looked up next to this script first, then in the current folder.

student_file_name = "student_datase0.csv"
job_file_name = "company_job_dataset_scrapped2809.csv"

random_seed = 42            # keeps the clustering result the same on every run
number_of_clusters = 3      # 3 student levels and 3 job tiers (low, mid, top)
tier_names = ["low", "mid", "top"]
level_names = ["beginner", "intermediate", "advanced"]

min_skill_match = 0.5       # a role is suggested only if the student covers 50% of its skills
roles_to_show = 8           # how many "apply now" roles to print
upgrade_roles_to_check = 10 # how many next-tier roles to study for the upgrade path

# soft skills are not matched like technical skills:
# "communication" is checked against the terminal answer, "problem solving" is
# assumed to be covered by the coding practice, so both are removed from matching.
soft_skills = {"communication", "problem solving"}

# common short forms students type, mapped to the names used in the job data
skill_aliases = {
    "ml": "machine learning", "dl": "deep learning", "ai": "machine learning",
    "oop": "object oriented programming", "oops": "object oriented programming",
    "dsa": "data structures", "ds": "data structures",
    "js": "javascript", "node": "node.js", "nodejs": "node.js",
    "k8s": "kubernetes", "cpp": "c++", "ci cd": "ci/cd", "cicd": "ci/cd",
    "sklearn": "scikit-learn", "powerbi": "power bi", "reactjs": "react",
    "sql server": "sql", "mysql": "sql", "postgres": "sql", "postgresql": "sql",
}

# communication answers converted to numbers (jobs asking for communication need 2 or more)
communication_scores = {"bad": 0, "low": 1, "mid": 2, "good": 3, "high": 3}

# sites asked one by one in the terminal, the counts are added together
coding_sites = ["leetcode", "hackerrank", "codechef", "codeforces", "other sites"]


def find_file(file_name):
    # look next to the script, then in the folder the script was started from
    script_folder = os.path.dirname(os.path.abspath(__file__))
    for folder in (script_folder, os.getcwd()):
        path = os.path.join(folder, file_name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"could not find {file_name}, keep it next to this script")



# section 2: small cleaning helpers
# everything is converted to lowercase so the two datasets and the terminal
# input all use exactly the same spelling.

def clean_skill(text):
    text = text.strip().lower()
    return skill_aliases.get(text, text)


def split_skills(text):
    # "python, SQL ,ml" -> ["machine learning", "python", "sql"]
    if pd.isna(text) or str(text).strip() == "":
        return []
    parts = [clean_skill(part) for part in str(text).split(",")]
    return sorted(set(part for part in parts if part))


def clean_degree(text):
    # "B.Tech" / "b tech" -> "btech", "B.E" -> "be"
    return str(text).strip().lower().replace(".", "").replace(" ", "")


def parse_experience(value):
    # dataset stores "fresher" or a number of years (assumed years, not months)
    value = str(value).strip().lower()
    if value == "fresher":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0



# section 3: load and prepare both datasets

students = pd.read_csv(find_file(student_file_name))
jobs = pd.read_csv(find_file(job_file_name))

# students: turn the skills text into a python list and count the skills
students["skill_list"] = students["technical_skills"].apply(split_skills)
students["skill_count"] = students["skill_list"].apply(len)

# jobs: same cleaning for skills, degrees and experience
jobs["required_list"] = jobs["required_skills"].apply(split_skills)
jobs["preferred_list"] = jobs["preferred_skills"].apply(split_skills)
jobs["degree_list"] = jobs["degrees_accepted"].apply(
    lambda text: [clean_degree(part) for part in str(text).split(",")]
)
jobs["exp_years"] = jobs["experience_required"].apply(parse_experience)

# every skill that appears anywhere in the two datasets (used to warn about unknown skills)
known_skills = set()
for skill_list in students["skill_list"]:
    known_skills.update(skill_list)
for skill_list in jobs["required_list"] + jobs["preferred_list"]:
    known_skills.update(skill_list)



# section 4: cluster the job roles into tiers (low, mid, top)
# each row is one company + role. rows with similar pay and hiring difficulty
# fall into the same cluster. features are scaled first so big numbers (salary)
# do not overpower small ones (cgpa), then salary is given 3x weight because
# a tier is mainly about pay. without this weight the tiers overlap heavily.

job_feature_columns = ["salary_avg_lpa", "min_cgpa", "exp_years", "coding_test_rounds"]
job_feature_weights = np.array([3, 1, 1, 1])
job_scaler = StandardScaler()
job_scaled = job_scaler.fit_transform(jobs[job_feature_columns].values) * job_feature_weights

job_model = KMeans(n_clusters=number_of_clusters, n_init=10, random_state=random_seed)
jobs["job_cluster"] = job_model.fit_predict(job_scaled)

# name the clusters by average salary: lowest salary cluster = low, highest = top
job_centers = job_scaler.inverse_transform(job_model.cluster_centers_ / job_feature_weights)
job_order = np.argsort(job_centers[:, 0])  # column 0 is salary
job_cluster_to_rank = {int(cluster_id): rank for rank, cluster_id in enumerate(job_order)}
jobs["tier_rank"] = jobs["job_cluster"].map(job_cluster_to_rank)
jobs["tier"] = jobs["tier_rank"].apply(lambda rank: tier_names[rank])



# section 5: cluster the students into levels (beginner, intermediate, advanced)
# students with similar cgpa, practice, projects, experience, certifications
# and skill count are grouped together. clusters are named by overall strength.

student_feature_columns = [
    "cgpa", "coding_questions_solved", "projects_count",
    "experience_months", "certifications_count", "skill_count",
]
student_scaler = StandardScaler()
student_scaled = student_scaler.fit_transform(students[student_feature_columns].values)

student_model = KMeans(n_clusters=number_of_clusters, n_init=10, random_state=random_seed)
students["student_cluster"] = student_model.fit_predict(student_scaled)

# strength of a cluster = average of its scaled feature values (higher = stronger students)
cluster_strength = student_model.cluster_centers_.mean(axis=1)
student_order = np.argsort(cluster_strength)
student_cluster_to_rank = {int(cluster_id): rank for rank, cluster_id in enumerate(student_order)}
students["level_rank"] = students["student_cluster"].map(student_cluster_to_rank)

# cluster centers in real units, used later to show gaps in the upgrade path
student_centers_real = student_scaler.inverse_transform(student_model.cluster_centers_)
center_by_rank = {student_cluster_to_rank[cid]: student_centers_real[cid] for cid in range(number_of_clusters)}

# the link between the two sides (no placement outcomes are available):
# a student level is linked to the job tier of the same rank, so beginner -> low,
# intermediate -> mid, advanced -> top. roles inside the tier are then chosen by skills.


def print_training_summary():
    # quick quality check for the report: silhouette score (closer to 1 is better)
    student_silhouette = silhouette_score(student_scaled, students["student_cluster"])
    job_silhouette = silhouette_score(job_scaled, jobs["job_cluster"])
    print("=" * 60)
    print("model training summary")
    print("=" * 60)
    print(f"students used: {len(students)} | jobs used: {len(jobs)}")
    print(f"silhouette score, students: {student_silhouette:.2f} | jobs: {job_silhouette:.2f}")
    print("\njob tiers found by clustering:")
    for rank, tier in enumerate(tier_names):
        part = jobs[jobs["tier_rank"] == rank]
        print(f"  {tier:<4} tier: {len(part):>3} roles, salary {part['salary_avg_lpa'].min():.1f} "
              f"to {part['salary_avg_lpa'].max():.1f} lpa (avg {part['salary_avg_lpa'].mean():.1f})")
    print("\nstudent levels found by clustering (average values):")
    for rank, level in enumerate(level_names):
        part = students[students["level_rank"] == rank]
        print(f"  {level:<12}: {len(part):>3} students, cgpa {part['cgpa'].mean():.1f}, "
              f"questions {part['coding_questions_solved'].mean():.0f}, "
              f"projects {part['projects_count'].mean():.1f}, "
              f"experience {part['experience_months'].mean():.1f} months")



# section 6: terminal input for one new student
# each helper keeps asking until the answer is valid.

def ask_text(prompt):
    while True:
        answer = input(prompt).strip().lower()
        if answer:
            return answer
        print("  please type something")


def ask_number(prompt, low, high, whole=False):
    while True:
        try:
            value = float(input(prompt).strip())
        except ValueError:
            print("  please type a number")
            continue
        if not low <= value <= high:
            print(f"  value must be between {low} and {high}")
            continue
        return int(value) if whole else value


def ask_optional_count(prompt):
    # blank answer means zero
    while True:
        answer = input(prompt).strip()
        if answer == "":
            return 0
        if answer.isdigit():
            return int(answer)
        print("  please type a whole number or press enter")


def get_student_input():
    print("\n" + "=" * 60)
    print("enter your details")
    print("=" * 60)
    degree = clean_degree(ask_text("degree (btech, be, bsc, bca, mtech): "))
    branch = ask_text("branch (example: computer science and engineering): ")
    cgpa = ask_number("cgpa (0 to 10): ", 0, 10)
    language = clean_skill(ask_text("primary coding language (example: python, c++, java): "))

    print("coding questions solved on each site (press enter for none):")
    solved = sum(ask_optional_count(f"  {site}: ") for site in coding_sites)

    skills = split_skills(ask_text("technical skills, comma separated (example: sql, ml, react): "))
    projects = ask_number("number of projects: ", 0, 50, whole=True)
    experience = ask_number("internship/work experience in months: ", 0, 240, whole=True)
    certifications = ask_number("number of certifications: ", 0, 50, whole=True)

    while True:
        comm_text = ask_text("communication skill (mid / low / bad): ")
        if comm_text in communication_scores:
            break
        print("  please type mid, low or bad")

    # the primary coding language is also a skill, so it is added to the skill list
    skills = sorted(set(skills + [language]))
    unknown = [skill for skill in skills if skill not in known_skills]
    if unknown:
        print(f"note: not found in our data, ignored for matching: {', '.join(unknown)}")

    return {
        "degree": degree, "branch": branch, "cgpa": cgpa, "language": language,
        "solved": solved, "skills": skills, "projects": projects,
        "experience": experience, "certifications": certifications,
        "comm_text": comm_text, "comm_score": communication_scores[comm_text],
    }



# section 7: match the student with the job side

def find_student_level(student):
    # scale the new student with the same scaler used in training, then find the nearest cluster
    row = np.array([[student["cgpa"], student["solved"], student["projects"],
                     student["experience"], student["certifications"], len(student["skills"])]])
    cluster_id = int(student_model.predict(student_scaler.transform(row))[0])
    return student_cluster_to_rank[cluster_id]


def score_roles(student):
    # for every job row: skill match (0 to 1) and a list of things blocking the application
    student_skill_set = set(student["skills"])
    rows = []
    for _, job in jobs.iterrows():
        required = [s for s in job["required_list"] if s not in soft_skills]
        preferred = [s for s in job["preferred_list"] if s not in soft_skills]
        matched = [s for s in required if s in student_skill_set]
        missing = [s for s in required if s not in student_skill_set]
        required_cover = len(matched) / len(required) if required else 1.0
        preferred_cover = (len([s for s in preferred if s in student_skill_set]) / len(preferred)
                           if preferred else 0.0)
        # required skills count 80%, preferred skills 20%
        skill_match = 0.8 * required_cover + 0.2 * preferred_cover

        # eligibility checks
        blockers = []
        if student["degree"] not in job["degree_list"]:
            blockers.append("degree not accepted")
        if student["cgpa"] < job["min_cgpa"]:
            blockers.append(f"cgpa below {job['min_cgpa']}")
        if student["experience"] < job["exp_years"] * 12:
            blockers.append(f"needs {int(job['exp_years'] * 12)} months experience")
        if "communication" in job["required_list"] and student["comm_score"] < 2:
            blockers.append("needs better communication")

        rows.append({
            "company": job["company_name"], "role": job["job_role"],
            "salary": job["salary_avg_lpa"], "tier_rank": int(job["tier_rank"]),
            "skill_match": skill_match, "matched": matched, "missing": missing,
            "blockers": blockers,
        })
    return pd.DataFrame(rows)


def print_role_line(item):
    print(f"  - {item['role']} at {item['company']} | {item['salary']:.1f} lpa | "
          f"skill match {item['skill_match'] * 100:.0f}%")
    print(f"      you have: {', '.join(item['matched']) or 'none'}")
    if item["missing"]:
        print(f"      missing : {', '.join(item['missing'])}")


def print_upgrade_path(student, scored, level_rank):
    print("\n" + "=" * 60)
    print("upgrade path")
    print("=" * 60)

    # aim for the next tier up; a top-level student works on the top tier itself
    if level_rank < number_of_clusters - 1:
        next_rank = level_rank + 1
        print(f"target: move from the {tier_names[level_rank]} tier towards the {tier_names[next_rank]} tier")
    else:
        next_rank = level_rank
        print("you are already in the top group; strengthen these to get more top-tier roles")

    # study the best skill-matching roles of the target tier, ignoring eligibility
    targets = scored[scored["tier_rank"] == next_rank].sort_values(
        ["skill_match", "salary"], ascending=False).head(upgrade_roles_to_check)

    if len(targets) > 0:
        print("\nbest-fit roles to aim for:")
        for _, item in targets.head(3).iterrows():
            print_role_line(item)

        # skills missing most often across those roles
        missing_counter = Counter(skill for skills in targets["missing"] for skill in skills)
        if missing_counter:
            print("\nskills to learn (most needed first):")
            for skill, count in missing_counter.most_common(5):
                print(f"  - {skill} (needed in {count} of {len(targets)} target roles)")

        # blockers that appear most often
        blocker_counter = Counter(text for blockers in targets["blockers"] for text in blockers)
        if blocker_counter:
            print("\nother things stopping you from those roles:")
            for text, count in blocker_counter.most_common(4):
                print(f"  - {text} ({count} of {len(targets)} roles)")

    # numeric gaps compared with the average student of the target level
    target_center = center_by_rank[next_rank]
    labels = ["cgpa", "coding questions solved", "projects", "experience months",
              "certifications", "number of skills"]
    yours = [student["cgpa"], student["solved"], student["projects"],
             student["experience"], student["certifications"], len(student["skills"])]
    # tiny gaps (under 5%) are ignored so only meaningful gaps are shown
    gaps = [(labels[i], yours[i], target_center[i]) for i in range(len(labels))
            if yours[i] < target_center[i] * 0.95]
    if gaps:
        print(f"\nyou are below the average {level_names[next_rank]} student in:")
        for label, mine, target in gaps:
            print(f"  - {label}: you {mine:.1f}, target about {target:.1f}")

    if student["comm_score"] < 2:
        print("\nimprove communication to at least 'mid', some roles need it")


def run_matching(student):
    level_rank = find_student_level(student)
    scored = score_roles(student)

    print("\n" + "=" * 60)
    print("result")
    print("=" * 60)
    print(f"your student group: {level_names[level_rank]}")
    print(f"target company tier: {tier_names[level_rank]}")

    # roles that can be applied for today: no blockers, enough skill match, target tier or lower
    ready = scored[(scored["blockers"].apply(len) == 0)
                   & (scored["skill_match"] >= min_skill_match)
                   & (scored["tier_rank"] <= level_rank)]
    ready = ready.sort_values(["tier_rank", "skill_match", "salary"], ascending=False)

    print(f"\nroles you can apply for now ({min(len(ready), roles_to_show)} of {len(ready)} shown):")
    if len(ready) == 0:
        print("  none yet, see the upgrade path below")
    for _, item in ready.head(roles_to_show).iterrows():
        print_role_line(item)
        print(f"      tier    : {tier_names[item['tier_rank']]}")

    print_upgrade_path(student, scored, level_rank)



# section 8: main program

def main():
    print_training_summary()
    while True:
        student = get_student_input()
        run_matching(student)
        again = input("\ncheck another student? (y/n): ").strip().lower()
        if again != "y":
            break
    print("done")


if __name__ == "__main__":
    main()
