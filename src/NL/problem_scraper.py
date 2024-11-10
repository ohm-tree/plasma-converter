# Scrapes problems from Art of Problem Solving.

import json
import os
import shutil
from xml.dom.minidom import Attr

import openai
import requests
import tqdm
from bs4 import BeautifulSoup
from pydantic import BaseModel

from src.utils.prompt import FewShotPrompter


class AopsMultipleChoiceProblem(BaseModel):
    latex_problem_statement: str
    latex_answer_choices: list[str]
    latex_correct_answer: str
    latex_solutions: list[str]
    problem_image_warning: bool
    solution_image_warnings: list[bool]


AMC_EXAMPLE_RAW_TEXT = (
    "{{duplicate|[[2007 AMC 12A Problems|2007 AMC 12A #1]] and [[2007 AMC 10A Problems/Problem 1|2007 AMC 10A #1]]}}\n"
    "== Problem ==\n"
    "One ticket to a show costs $\\$$20$ at full price. Susan buys 4 tickets using a coupon that gives her a 25% discount. Pam buys 5 tickets using a coupon that gives her a 30% discount. How many more dollars does Pam pay than Susan?\n\n"
    "$\\mathrm{(A)}\\ 2\\qquad \\mathrm{(B)}\\ 5\\qquad \\mathrm{(C)}\\ 10\\qquad \\mathrm{(D)}\\ 15\\qquad \\mathrm{(E)}\\ 20$\n\n"
    "== Official Solution ==\n"
    "$\\textbf{Answer: (C)}$ \n"
    "Susan pays $(4)(0.75)(20) = 60$ dollars. Pam pays $(5)(0.70)(20) = 70$ dollars, so she pays $70-60=10$ more dollars than Susan.\n\n"
    "== See also ==\n"
    "{{AMC12 box|year=2007|ab=A|before=First question|num-a=2}}\n"
    "{{AMC10 box|year=2007|ab=A|before=First question|num-a=2}}\n\n"
    "[[Category:Introductory Algebra Problems]]\n"
    "{{MAA Notice}}\n"
)
AMC_EXAMPLE_RESPONSE_JSON = {
    "latex_problem_statement": (
        "One ticket to a show costs $\\$20$ at full price. Susan buys 4 tickets using a coupon that gives her a $25\\%$ discount. "
        "Pam buys $5$ tickets using a coupon that gives her a $30\\%$ discount. How many more dollars does Pam pay than Susan?"
    ),
    "latex_answer_choices": ["$2$", "$5$", "$10$", "$15$", "$20$"],
    "latex_correct_answer": "C",
    "latex_solutions": [
        "Susan pays $(4)(0.75)(20) = 60$ dollars. Pam pays $(5)(0.70)(20) = 70$ dollars, so she pays $70-60=10$ more dollars than Susan."
    ],
    "problem_image_warning": False,
    "solution_image_warnings": [False]
}
AMC_EXAMPLE_RESPONSE_JSON_STR = json.dumps(AMC_EXAMPLE_RESPONSE_JSON)

prompter = FewShotPrompter(
    fstring_prompt=(
        "Parse the following text into a JSON object.\n"
        "{raw_text}"
    ),
    system_prompt=(
        "You perform data processing. "
        "You are given a string of raw mathbb problem and solution text. "
        "Extract the problem statement, answer choices, correct answer, "
        "and solutions from the text. Rewrite all mathbb expressions as LaTeX.\n"
        "Sometimes, there may be a key image in the problem statement."
        "If it is impossible to solve the problem without the image, "
        "then you should return `True` for `problem_image_warning`.\n"
        "Similarly, sometimes there may be a key image in one or more "
        "of the solutions. If it is impossible to solve the problem "
        "without the image, then you should return `True` for "
        "`solution_image_warning`.\n"
        "The answer choice should be a single letter, such as `A` or `B`."
    ),
    few_shot_examples=[
        {
            "raw_text": AMC_EXAMPLE_RAW_TEXT
        }
    ],
    few_shot_responses=[
        AMC_EXAMPLE_RESPONSE_JSON_STR
    ]
)


def scrape_amc_problem(url: str) -> AopsMultipleChoiceProblem:
    """
    Consumes a URL like https://artofproblemsolving.com/wiki/index.php?title=2007_AMC_12A_Problems/Problem_1&action=edit.

    OpenAI gpt-4o-mini is used to extract the problem statement and solutions.
    """
    try:
        response = requests.get(url)

    except requests.exceptions.RequestException as e:
        print(f"\nError fetching {url}: {e}\n")
        return None
    # On aops wiki edit pages, the main body of the page is inside a div with class "mw-editfont-monospace".
    try:
        soup = BeautifulSoup(response.text, "html.parser")
        main_body = soup.find("textarea", {"class": "mw-editfont-monospace"})
        # Just take EVERYTHING from this div and give it to gpt-4o-mini.
        text = main_body.text
    except AttributeError as e:
        print(f"\nError parsing {url}: {e}\n")
        return None

    # if the text starts with #redirect, then it is a duplicate.
    if text.startswith("#redirect"):
        print(f"\n{url} is a duplicate.\n")
        return None

    # replace all <math> and </math> with $$.
    text = text.replace("<math>", "$").replace("</math>", "$")
    # # Use gpt-4o-mini to extract the problem statement and solutions.

    prompt = prompter.few_shot_prompt(raw_text=main_body.text)
    response = openai.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=prompt,
        response_format=AopsMultipleChoiceProblem
    )
    return response.choices[0].message.parsed


def scrape_amc(
        year: int,
        level: str,
        problem_number: int,
        refresh: bool = False
) -> AopsMultipleChoiceProblem:
    """
    Scrapes an AMC problem and dumps the result in a JSON file.
    """
    filepath = f"src/NL/data/amc/{year}/{level}/{problem_number}/problem.json"

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Check if the file already exists
    if os.path.exists(filepath) and not refresh:
        with open(filepath, "r") as f:
            res = json.load(f)
            if res:
                return res
            return None

    url = f"https://artofproblemsolving.com/wiki/index.php?title={year}_AMC_{level}_Problems/Problem_{problem_number}&action=edit"
    res = scrape_amc_problem(url)
    if res is None:
        with open(filepath, "w") as f:
            json.dump({}, f)
        return None
    res = res.model_dump()
    # dump the result in json to a file
    with open(filepath, "w") as f:
        json.dump(res, f)
    return res


if __name__ == "__main__":
    # Scrape every AMC problem from 2024.
    with tqdm.tqdm(total=4 * 25) as pbar:
        for level in ["10A", "10B", "12A", "12B"]:
            for problem_number in range(1, 25):
                res = scrape_amc(2024, level, problem_number)
                pbar.update(1)
                pbar.set_postfix(level=level, problem_number=problem_number)
