import unittest
from unittest.mock import patch

from handlers.ToolHandle import ToolHandle
from tools.MathTools import (
    AVAILABLE_MATH_TOOLS,
    calculate_limit,
    calculate_expression,
    differentiate_expression,
    integrate_expression,
    register_math_tools,
    solve_equation,
)


class MathToolsTests(unittest.TestCase):
    def test_arithmetic_and_exact_result(self):
        result = calculate_expression("(2 + 3) * 4 - 6 / 3")

        self.assertTrue(result["success"])
        self.assertEqual(result["result"], "18")
        self.assertEqual(result["approximation"], "18.0000000000000")

    def test_functions_and_constants(self):
        result = calculate_expression("sqrt(2) + sin(pi / 2)", precision=8)

        self.assertTrue(result["success"])
        self.assertEqual(result["result"], "1 + sqrt(2)")
        self.assertEqual(result["approximation"], "2.4142136")

    def test_division_by_zero_returns_structured_error(self):
        result = calculate_expression("1 / (2 - 2)")

        self.assertFalse(result["success"])
        self.assertIn("zero", result["error"])

    def test_invalid_syntax_returns_structured_error(self):
        result = calculate_expression("2 +")

        self.assertFalse(result["success"])
        self.assertIn("składnia", result["error"])

    def test_unsupported_python_syntax_is_rejected(self):
        result = calculate_expression("__import__('os').system('echo unsafe')")

        self.assertFalse(result["success"])
        self.assertIn("niedozwoloną składnię", result["error"])

    def test_eval_is_rejected_as_an_unknown_function(self):
        result = calculate_expression("eval('1 + 1')")

        self.assertFalse(result["success"])
        self.assertIn("nie jest dozwolona", result["error"])

    def test_unknown_math_function_is_rejected(self):
        result = calculate_expression("made_up_function(2)")

        self.assertFalse(result["success"])
        self.assertIn("nie jest dozwolona", result["error"])

    def test_advanced_special_functions(self):
        result = calculate_expression(
            "log10(100) + factorial(5) + binomial(5, 2) + gamma(5)"
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["result"], "156")

    def test_differentiation(self):
        result = differentiate_expression("x^3 + sin(x)")

        self.assertTrue(result["success"])
        self.assertEqual(result["result"], "3*x**2 + cos(x)")

    def test_definite_integration(self):
        result = integrate_expression("x^2", lower_bound="0", upper_bound="1")

        self.assertTrue(result["success"])
        self.assertEqual(result["result"], "1/3")

    def test_limit(self):
        result = calculate_limit("sin(x)/x")

        self.assertTrue(result["success"])
        self.assertEqual(result["result"], "1")

    def test_power_and_precision_limits(self):
        self.assertTrue(calculate_expression("2^10")["success"])
        too_large = calculate_expression("2^101")
        invalid_precision = calculate_expression("pi", precision=100)

        self.assertFalse(too_large["success"])
        self.assertFalse(invalid_precision["success"])

    def test_calculation_timeout_returns_structured_error(self):
        with patch("tools.MathTools.CALCULATION_TIMEOUT_SECONDS", 0):
            result = calculate_expression("1 + 1")

        self.assertFalse(result["success"])
        self.assertIn("Przekroczono limit czasu", result["error"])

    def test_solves_quadratic_equation(self):
        result = solve_equation("x^2 - 9 = 0")

        self.assertTrue(result["success"])
        self.assertEqual(set(result["solutions"]), {"-3", "3"})

    def test_equation_with_no_solution(self):
        result = solve_equation("x + 1 = x + 2")

        self.assertTrue(result["success"])
        self.assertEqual(result["solutions"], [])
        self.assertEqual(result["message"], "Brak rozwiązań.")

    def test_periodic_equation_returns_infinite_solution_set(self):
        result = solve_equation("sin(x) = 0")

        self.assertTrue(result["success"])
        self.assertIn("ImageSet", result["solutions"][0])

    def test_identity_equation_returns_all_values(self):
        result = solve_equation("x = x")

        self.assertTrue(result["success"])
        self.assertEqual(result["solutions"], ["wszystkie wartości x"])

    def test_equation_syntax_validation(self):
        result = solve_equation("x + = 2")

        self.assertFalse(result["success"])
        self.assertIn("składnia", result["error"])

    def test_tool_definitions_register_with_tool_handle(self):
        handle = ToolHandle()
        register_math_tools(handle)

        self.assertEqual(
            {tool["function"]["name"] for tool in handle.get_tools_list()},
            set(AVAILABLE_MATH_TOOLS),
        )
        self.assertIn("expression", handle.get_tool("calculate_expression")["function"]["parameters"]["required"])
        self.assertIn("equation", handle.get_tool("solve_equation")["function"]["parameters"]["required"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
