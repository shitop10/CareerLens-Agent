import unittest

from app.core import prompts


class TestPlannerPrompt(unittest.TestCase):
    def test_has_tools_placeholder(self):
        self.assertIn("{tools}", prompts.planner_system_prompt)

    def test_formats_with_four_tool_names(self):
        rendered = prompts.planner_system_prompt.format(tools="- hybrid_retrieval：检索")
        self.assertIn("hybrid_retrieval", rendered)
        for name in ("memory_lookup", "resume_diagnosis", "job_match"):
            self.assertIn(name, rendered)

    def test_forbids_hallucinated_tools(self):
        self.assertIn("禁止发明工具名", prompts.planner_system_prompt)

    def test_mandates_retrieval(self):
        self.assertIn("必须调用 hybrid_retrieval", prompts.planner_system_prompt)

    def test_caps_tool_count(self):
        self.assertIn("最多调用 3 个工具", prompts.planner_system_prompt)

    def test_contains_rewrite_example(self):
        self.assertIn("改写示例", prompts.planner_system_prompt)


class TestReflectorPrompt(unittest.TestCase):
    def test_placeholders(self):
        self.assertIn("{query}", prompts.reflector_prompt)
        self.assertIn("{evidence}", prompts.reflector_prompt)

    def test_forbids_fabrication(self):
        self.assertIn("禁止补充任何未出现的证据", prompts.reflector_prompt)

    def test_formats_cleanly(self):
        rendered = prompts.reflector_prompt.format(query="Q", evidence="E")
        self.assertIn("Q", rendered)
        self.assertIn("E", rendered)


class TestJudgePrompt(unittest.TestCase):
    def test_placeholders(self):
        for key in ("{query}", "{evidence}", "{answer}"):
            self.assertIn(key, prompts.judge_prompt)

    def test_requires_total_line(self):
        self.assertIn("TOTAL=", prompts.judge_prompt)

    def test_formats_cleanly(self):
        rendered = prompts.judge_prompt.format(query="Q", evidence="E", answer="A")
        self.assertNotIn("{", rendered)


class TestExistingPromptsUntouched(unittest.TestCase):
    def test_rag_system_prompt_still_present(self):
        self.assertIn("CareerLens", prompts.rag_system_prompt)
        self.assertIn("{context}", prompts.rag_system_prompt)

    def test_title_and_summary_prompts_intact(self):
        self.assertIn("{user_input}", prompts.title_generation_prompt)
        self.assertIn("{content}", prompts.summary_generation_prompt)


if __name__ == "__main__":
    unittest.main()
