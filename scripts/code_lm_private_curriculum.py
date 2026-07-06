"""Private Code LM curriculum construction and governed train-row loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from code_lm_private_rows import DIAGNOSTIC_ONLY_HIGH_TRANSFER_PRIVATE_ROWS
from progress_integrity_policy import is_non_promotable_diagnostic_concept

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_ONLY_PRIVATE_ROW_PATHS = {
    str(Path(path)).replace("\\", "/").lower()
    for path in DIAGNOSTIC_ONLY_HIGH_TRANSFER_PRIVATE_ROWS
}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def safe_name(value: Any) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value or "item")).strip("_") or "item"


def build_private_curriculum(*, seed: int, count: int) -> list[dict[str, Any]]:
    categories = [
        ("opposite_signs", "Return whether two numbers have opposite signs.", "(data < 0) != (other < 0)", [(1, -2, True), (-3, -2, False), (0, 4, False)]),
        ("sum_list", "Return the sum of a list of integers.", "sum(data)", [([1, 2, 3], None, 6), ([], None, 0), ([-1, 4], None, 3)]),
        ("max_list", "Return the maximum item in a non-empty list.", "max(data)", [([1, 5, 2], None, 5), ([-3, -1], None, -1), ([7], None, 7)]),
        ("reverse_string", "Return the reverse of a string.", "data[::-1]", [("abc", None, "cba"), ("", None, ""), ("level", None, "level")]),
        ("one_less_than_twice_reverse", "Return whether an integer is one less than twice its digit reversal.", "data == 2 * int(str(abs(data))[::-1]) - 1", [(73, None, True), (70, None, False), (1, None, True)]),
        ("palindrome", "Return whether a string reads the same backward.", "data == data[::-1]", [("level", None, True), ("abc", None, False), ("", None, True)]),
        ("median_list", "Return the median of a numeric list, using the average for even-length lists.", "", [([3, 1, 2, 4, 5], None, 3), ([-10, 4, 6, 1000, 10, 20], None, 8.0), ([2, 8], None, 5.0)]),
        ("modular_power_two", "Return two raised to n modulo the provided modulus.", "", [(3, 5, 3), (0, 101, 1), (6, 7, 1)]),
        ("caesar_decode_shift5", "Decode a lowercase alphabetic string that was shifted forward by five letters.", "", [("fgh", None, "abc"), ("", None, ""), ("mjqqt", None, "hello")]),
        ("remove_vowels", "Return the string with vowels removed regardless of case, preserving all non-vowel characters.", "", [("abcdef", None, "bcdf"), ("aaBAA", None, "B"), ("zbcd", None, "zbcd")]),
        ("below_threshold", "Return whether every number in a list is strictly below a threshold.", "", [([1, 2, 4, 10], 100, True), ([1, 20, 4, 10], 5, False), ([], 0, True)]),
        ("add_numbers", "Return the sum of two numbers.", "data + other", [(2, 3, 5), (5, 7, 12), (-2, 8, 6)]),
        ("same_chars", "Return whether two strings contain the same set of characters.", "", [("abcd", "dddddddabc", True), ("eabcd", "dddddddabc", False), ("", "", True)]),
        ("count_vowels", "Return how many vowels are in a string; final y also counts as a vowel.", "", [("abc", None, 1), ("sky", None, 1), ("AEIOU", None, 5), ("bYe", None, 1)]),
        ("abs_diff", "Return the absolute difference between two numbers.", "abs(data - other)", [(7, 2, 5), (2, 7, 5), (-1, 4, 5)]),
        ("all_prefixes", "Return all prefixes of a string from shortest to longest.", "[data[:idx] for idx in range(1, len(data) + 1)]", [("", None, []), ("abc", None, ["a", "ab", "abc"]), ("x", None, ["x"])]),
        ("string_sequence", "Return a string containing numbers from 0 through n separated by spaces.", "' '.join(str(idx) for idx in range(data + 1))", [(0, None, "0"), (3, None, "0 1 2 3"), (5, None, "0 1 2 3 4 5")]),
        ("largest_concat", "Arrange a list of positive integers to form the largest concatenated integer.", "int(''.join(sorted((str(item) for item in data), key=lambda item: item * 8, reverse=True)))", [([1, 2, 3], None, 321), ([10, 2], None, 210), ([9, 91], None, 991)]),
        ("even_number", "Return whether a number is even.", "data % 2 == 0", [(2, None, True), (3, None, False), (0, None, True)]),
        ("length", "Return the length of a collection or string.", "len(data)", [("", None, 0), ("abc", None, 3), ([1, 2], None, 2)]),
        ("distinct_count", "Return the number of distinct items or characters.", "len(set(data))", [("", None, 0), ("banana", None, 3), ([1, 1, 2], None, 2)]),
        ("filter_integers", "Return only the integer items from a list.", "[item for item in data if isinstance(item, int)]", [([1, "x", 2.0, 3], None, [1, 3]), ([], None, []), (["a"], None, [])]),
        ("min_list", "Return the smallest item in a non-empty list.", "min(data)", [([10, 20, 1], None, 1), ([-2, 4], None, -2), ([7], None, 7)]),
        ("min_three", "Return the smallest of three scalar values.", "min(data, other, extra[0])", [({"__args__": (10, 20, 1)}, None, 1), ({"__args__": (-2, 4, -9)}, None, -9), ({"__args__": (7, 7, 8)}, None, 7)]),
        ("string_odd_index_remove", "Return a string with characters at odd indices removed.", "data[::2]", [("abcdef", None, "ace"), ("python", None, "pto"), ("", None, "")]),
        ("replace_whitespace", "Replace blank spaces in text with the provided character.", "data.replace(' ', other)", [("a b c", "-", "a-b-c"), ("no_space", "*", "no_space"), (" ", "_", "_")]),
        ("stable_negative_partition", "Stable-partition the first n list items so negatives come before non-negatives.", "", [({"__args__": ([-1, 2, -3, 4, 5], 5)}, None, [-1, -3, 2, 4, 5]), ({"__args__": ([1, -2, 3, -4, 9], 4)}, None, [-2, -4, 1, 3, 9]), ({"__args__": ([1, 2], 2)}, None, [1, 2])]),
        ("top_k_largest", "Return the n largest items from a list.", "sorted(data, reverse=True)[:other]", [([10, 20, 50, 70, 90], 2, [90, 70]), ([1, 1, 2], 5, [2, 1, 1]), ([], 3, [])]),
        ("cube_volume", "Return the volume of a cube from its side length.", "data * data * data", [(3, None, 27), (1, None, 1), (0, None, 0)]),
        ("cube_lateral_surface_area", "Return the lateral surface area of a cube from its side length.", "4 * data * data", [(5, None, 100), (3, None, 36), (1, None, 4)]),
        ("cylinder_lateral_surface_area", "Return the lateral surface area of a cylinder from radius and height.", "2 * 3.141592653589793 * data * other", [(1, 1, 6.283185307179586), (2, 5, 62.83185307179586), (0, 3, 0.0)]),
        ("string_char_count", "Return the number of characters in a string.", "len(data)", [("abc", None, 3), ("", None, 0), ("a b", None, 3)]),
        ("nonempty_substring_count", "Return the number of non-empty substrings of a string.", "len(data) * (len(data) + 1) // 2", [("abc", None, 6), ("a", None, 1), ("", None, 0)]),
        ("list_tail_replace", "Replace the final item of the first list with all items from the second list.", "data[:-1] + list(other)", [([1, 2, 3], [4, 5], [1, 2, 4, 5]), ([1], [9], [9]), ([], [1], [1])]),
        ("tuple_frequency_dict", "Return a dictionary counting occurrences of each tuple in a list.", "", [([(1, 2), (1, 2), (3, 4)], None, {(1, 2): 2, (3, 4): 1}), ([], None, {}), ([("a", "b")], None, {("a", "b"): 1})]),
        ("tuple_item_count", "Return how many times an item occurs in a tuple or sequence.", "data.count(other)", [((1, 2, 1), 1, 2), (("a",), "b", 0), ((3, 3, 3), 3, 3)]),
        ("count_integer_items", "Return how many elements in a list are exact integers.", "sum(1 for item in data if isinstance(item, int))", [([1, "x", 2.0, 3], None, 2), ([True, 4, "5"], None, 2), ([], None, 0)]),
        ("split_list_at_index", "Split a list at an index and return the two parts.", "(data[:other], data[other:])", [([1, 2, 3, 4], 2, ([1, 2], [3, 4])), ([1], 0, ([], [1])), ([], 3, ([], []))]),
        ("swap_pair", "Return a pair with the two provided values swapped.", "(other, data)", [(1, 2, (2, 1)), ("a", "b", ("b", "a")), (0, 0, (0, 0))]),
        ("tuple_elementwise_division", "Divide matching elements of two numeric tuples.", "tuple(left / right for left, right in zip(data, other))", [((10, 20), (2, 5), (5.0, 4.0)), ((9,), (3,), (3.0,)), ((), (), ())]),
        ("tuple_elementwise_max", "Return element-wise maxima for two tuples.", "tuple(max(left, right) for left, right in zip(data, other))", [((1, 5), (2, 3), (2, 5)), ((9,), (3,), (9,)), ((), (), ())]),
        ("tuple_nested_elementwise_max", "Return element-wise maxima for matching nested tuple pairs.", "", [(((1, 3), (4, 5)), ((6, 7), (3, 9)), ((6, 7), (4, 9))), (((2, 9),), ((1, 1),), ((2, 9),)), ((), (), ())]),
        ("insert_before_each", "Insert an item before every item in a list.", "", [([1, 2, 3], 0, [0, 1, 0, 2, 0, 3]), ([], "x", []), (["a"], "z", ["z", "a"])]),
        ("count_primes_below", "Return how many prime numbers are strictly below n.", "", [(10, None, 4), (0, None, 0), (20, None, 8)]),
        ("next_perfect_square", "Return the next perfect square greater than a number.", "", [(10, None, 16), (25, None, 36), (0, None, 1)]),
        ("harmonic_sum", "Return the harmonic sum through n terms.", "sum(1 / item for item in range(1, data + 1))", [(1, None, 1.0), (2, None, 1.5), (4, None, 2.083333333333333)]),
        ("list_chunks_every_n", "Split a list into consecutive chunks of length n.", "[data[idx:idx + other] for idx in range(0, len(data), other)]", [([1, 2, 3, 4, 5], 2, [[1, 2], [3, 4], [5]]), ([1, 2], 5, [[1, 2]]), ([], 3, [])]),
        ("combinations_with_replacement", "Return all combinations with repetition of length n from the input list.", "list(itertools.combinations_with_replacement(data, other))", [([1, 2], 2, [(1, 1), (1, 2), (2, 2)]), (["a"], 3, [("a", "a", "a")]), ([], 2, [])]),
        ("sort_by_second", "Sort pairs by the second value.", "sorted(data, key=lambda item: item[1])", [([("b", 2), ("a", 1)], None, [("a", 1), ("b", 2)]), ([], None, []), ([("x", 3)], None, [("x", 3)])]),
        ("max_tuple_difference", "Return the largest absolute difference inside a list of pairs.", "max(abs(left - right) for left, right in data)", [([(3, 5), (1, 7)], None, 6), ([(10, 3)], None, 7), ([(0, 0), (-2, 3)], None, 5)]),
        ("positive_count", "Return how many numbers in a list are positive.", "sum(1 for item in data if item > 0)", [([1, -2, 3], None, 2), ([-1, 0], None, 0), ([5], None, 1)]),
        ("negative_count", "Return how many numbers in a list are negative.", "sum(1 for item in data if item < 0)", [([1, -2, 3], None, 1), ([-1, 0, -4], None, 2), ([5], None, 0)]),
        ("substring_count", "Return how many times a substring occurs in a string.", "data.count(other)", [("aaaa", "aa", 2), ("abc", "x", 0), ("banana", "na", 2)]),
        ("normalize_string", "Return a lowercase stripped version of a string.", "data.strip().lower()", [("  ABC ", None, "abc"), ("x", None, "x"), ("", None, "")]),
        ("safe_head", "Return the first list item or a fallback value when empty.", "data[0] if data else other", [([1, 2], 0, 1), ([], "none", "none"), (["x"], "none", "x")]),
        ("dict_required_keys", "Return whether a dictionary contains all required keys.", "all(key in data for key in other)", [({"a": 1, "b": 2}, ["a"], True), ({"a": 1}, ["a", "b"], False), ({}, [], True)]),
        ("public_private_count", "Return the number of public test cases in a mapping.", "len(data.get('public_test_cases', []))", [({"public_test_cases": [1, 2], "private_test_cases": [3]}, None, 2), ({}, None, 0), ({"public_test_cases": []}, None, 0)]),
        ("stable_dedupe", "Return list items with duplicates removed while preserving order.", "list(dict.fromkeys(data))", [(["io", "math", "io"], None, ["io", "math"]), ([], None, []), ([1, 1, 2], None, [1, 2])]),
        ("largest_divisor", "Return the largest proper divisor of an integer, or 1 for primes.", "max([item for item in range(1, data) if data % item == 0] or [1])", [(3, None, 1), (10, None, 5), (21, None, 7)]),
        ("factors", "Return all factors of a positive integer.", "[item for item in range(1, data + 1) if data % item == 0]", [(1, None, [1]), (6, None, [1, 2, 3, 6]), (7, None, [1, 7])]),
        ("prime_factors", "Return prime factors of an integer with multiplicity.", "", [(12, None, [2, 2, 3]), (7, None, [7]), (1, None, [])]),
        ("divisible_by_11", "Return whether an integer is evenly divisible by eleven.", "data % 11 == 0", [(121, None, True), (22, None, True), (25, None, False)]),
        ("nested_sum", "Return the sum of a list that may contain nested lists.", "sum((sum(item) if isinstance(item, list) else item) for item in data)", [([1, [2, 3]], None, 6), ([], None, 0), ([[1, 2], [3]], None, 6)]),
        ("rescale_to_unit", "Rescale a list of numbers to the 0 to 1 interval.", "[(item - min(data)) / (max(data) - min(data)) for item in data]", [([2.0, 4.0], None, [0.0, 1.0]), ([1, 2, 3], None, [0.0, 0.5, 1.0]), ([-1, 1], None, [0.0, 1.0])]),
        ("decode_cyclic", "Decode text encoded by rotating each complete three-character group left by one.", "", [("bca", None, "abc"), ("bcaefdhig", None, "abcdefghi"), ("ab", None, "ab")]),
        ("prime_fib_sequence", "Return the nth Fibonacci number that is also prime, counting from one.", "", [(1, None, 2), (2, None, 3), (3, None, 5), (5, None, 89)]),
        ("polynomial_zero_bisection", "Find one real zero of a polynomial represented by coefficient list.", "", [([1, 2], None, -0.5), ([-6, 11, -6, 1], None, 1.0)]),
        ("closest_pair", "Return the closest pair of numbers from a list.", "min(((left, right) for idx, left in enumerate(data) for right in data[idx + 1:]), key=lambda pair: abs(pair[0] - pair[1]))", [([1.0, 2.0, 2.2], None, (2.0, 2.2)), ([3, 8, 5], None, (3, 5)), ([10, 9], None, (10, 9))]),
        ("sum_squares", "Return the sum of squared numbers in a list.", "sum(item * item for item in data)", [([1, 2, 3], None, 14), ([], None, 0), ([-2, 3], None, 13)]),
        ("average_or_zero", "Return the average of a list, or zero for an empty list.", "sum(data) / len(data) if data else 0", [([2, 4], None, 3.0), ([], None, 0), ([5], None, 5.0)]),
        ("median_odd", "Return the median of an odd-length list.", "sorted(data)[len(data) // 2]", [([3, 1, 2], None, 2), ([9], None, 9), ([5, 1, 7, 3, 9], None, 5)]),
        ("gcd_pair", "Return the greatest common divisor of two positive integers.", "max(item for item in range(1, min(abs(data), abs(other)) + 1) if data % item == 0 and other % item == 0)", [(12, 18, 6), (7, 13, 1), (21, 14, 7)]),
        ("is_prime", "Return whether a positive integer is prime.", "data > 1 and all(data % item != 0 for item in range(2, int(data ** 0.5) + 1))", [(2, None, True), (9, None, False), (17, None, True)]),
        ("powers_of_two", "Return the first n powers of two.", "[2 ** idx for idx in range(data)]", [(0, None, []), (4, None, [1, 2, 4, 8]), (1, None, [1])]),
        ("flatten_once", "Flatten a list by one level.", "[child for item in data for child in (item if isinstance(item, list) else [item])]", [([[1, 2], 3], None, [1, 2, 3]), ([], None, []), ([1, [2], [3, 4]], None, [1, 2, 3, 4])]),
        ("word_count", "Return how many whitespace-separated words are in a string.", "len(data.split())", [("hello world", None, 2), ("", None, 0), (" one  two ", None, 2)]),
        ("remove_spaces", "Return the string with spaces removed.", "data.replace(' ', '')", [("a b c", None, "abc"), ("no", None, "no"), (" ", None, "")]),
        ("title_case_words", "Capitalize every word in a string.", "' '.join(word.capitalize() for word in data.split())", [("hello world", None, "Hello World"), ("x", None, "X"), ("", None, "")]),
        ("is_anagram", "Return whether two strings are anagrams.", "sorted(data) == sorted(other)", [("listen", "silent", True), ("abc", "abd", False), ("", "", True)]),
        ("common_elements", "Return sorted common unique elements from two lists.", "sorted(set(data) & set(other))", [([3, 1, 2], [2, 4, 1], [1, 2]), ([], [1], []), ([5], [5], [5])]),
        ("list_difference", "Return items from the first list that are not in the second list.", "[item for item in data if item not in other]", [([1, 2, 3], [2], [1, 3]), ([], [1], []), (["a", "b"], ["c"], ["a", "b"])]),
        ("transpose_matrix", "Transpose a rectangular matrix.", "[list(row) for row in zip(*data)]", [([[1, 2], [3, 4]], None, [[1, 3], [2, 4]]), ([[1], [2]], None, [[1, 2]]), ([[1, 2, 3]], None, [[1], [2], [3]])]),
        ("dot_product", "Return the dot product of two number lists.", "sum(left * right for left, right in zip(data, other))", [([1, 2, 3], [4, 5, 6], 32), ([], [], 0), ([2], [8], 16)]),
        ("clamp_number", "Clamp a number into an inclusive range passed as a pair.", "max(other[0], min(data, other[1]))", [(5, (1, 10), 5), (-2, (0, 3), 0), (9, (0, 3), 3)]),
        ("parse_ints", "Parse all integer-looking tokens from a string.", "[int(part) for part in data.split() if part.lstrip('-').isdigit()]", [("1 two -3", None, [1, -3]), ("none", None, []), ("4 5", None, [4, 5])]),
        ("symbol_beat_parser", "Parse whitespace-separated note symbols into beat counts using o=4, o|=2, and .|=1.", "[{'o': 4, 'o|': 2, '.|': 1}[part] for part in data.split()]", [("o o| .|", None, [4, 2, 1]), ("", None, []), (".| o", None, [1, 4])]),
        ("take_every_other", "Return every other item from a sequence.", "data[::2]", [([1, 2, 3, 4], None, [1, 3]), ("abcd", None, "ac"), ([], None, [])]),
        ("remove_none", "Return a list with None values removed.", "[item for item in data if item is not None]", [([1, None, 2], None, [1, 2]), ([None], None, []), ([], None, [])]),
        ("index_or_minus_one", "Return the index of an item, or -1 when absent.", "data.index(other) if other in data else -1", [([1, 2, 3], 2, 1), (["a"], "b", -1), ("hello", "l", 2)]),
        ("count_truthy", "Return how many items are truthy.", "sum(1 for item in data if item)", [([0, 1, "", "x"], None, 2), ([], None, 0), ([True, False, True], None, 2)]),
        ("matrix_diagonal", "Return the main diagonal of a rectangular matrix.", "[data[idx][idx] for idx in range(min(len(data), len(data[0]) if data else 0))]", [([[1, 2], [3, 4]], None, [1, 4]), ([[1, 2, 3]], None, [1]), ([], None, [])]),
        ("extract_def_name", "Return the first Python function name found in source text, or an empty string when absent.", "", [("def add(a, b):\n    return a + b\n", None, "add"), ("import math\n\ndef solve(nums):\n    pass\n", None, "solve"), ("x = 1", None, "")]),
        ("sorted_unique_values", "Return unique values from a collection in sorted order.", "sorted(set(data))", [([3, 1, 3, 2], None, [1, 2, 3]), ([], None, []), ([5, 5], None, [5])]),
        ("sort_even_index_values", "Sort values at even positions while leaving odd positions unchanged.", "", [([5, 9, 1, 8, 3], None, [1, 9, 3, 8, 5]), ([2, 1], None, [2, 1]), ([], None, [])]),
        ("increment_each_item", "Return a list where each numeric item is incremented by one.", "[item + 1 for item in data]", [([1, 2, 3], None, [2, 3, 4]), ([], None, []), ([-1, 0], None, [0, 1])]),
        ("count_digit_under_divisibility", "Count occurrences of a digit in numbers below a limit that satisfy either divisor.", "", [(40, (5, 7, 3), 2), (10, (2, 3, 9), 1), (100, (11, 13, 7), 3)]),
        ("two_sum_zero_exists", "Return whether any two distinct items sum to zero.", "", [([1, -1, 2], None, True), ([1, 2, 3], None, False), ([], None, False)]),
        ("three_sum_zero_exists", "Return whether any three distinct positions sum to zero.", "", [([-1, 0, 1], None, True), ([1, 2, 3], None, False), ([0, 0], None, False)]),
        ("base_digits", "Return the representation of a non-negative integer in a small base.", "", [(0, 2, "0"), (6, 2, "110"), (31, 5, "111")]),
        ("triangle_area_product", "Return half the product of a base and a height.", "data * other / 2", [(5, 3, 7.5), (10, 2, 10.0), (0, 7, 0.0)]),
        ("triangle_area_sides", "Return the area of a valid three-sided triangle rounded to two decimals, or -1 when invalid.", "", [({"__args__": (5, 5, 6)}, None, 12.0), ({"__args__": (2, 2, 5)}, None, -1), ({"__args__": (6, 8, 10)}, None, 24.0)]),
        ("balanced_brackets_simple", "Return whether angle-free bracket text is balanced.", "", [("([])", None, True), ("([)]", None, False), ("", None, True)]),
        ("monotonic_sequence", "Return whether a numeric sequence is nondecreasing or nonincreasing.", "", [([1, 2, 2, 3], None, True), ([3, 2, 1], None, True), ([1, 3, 2], None, False)]),
        ("largest_prime_factor", "Return the largest prime factor of a positive integer greater than one.", "", [(10, None, 5), (27, None, 3), (17, None, 17)]),
        ("arithmetic_series_sum", "Return the sum of all integers from zero through n.", "", [(0, None, 0), (4, None, 10), (7, None, 28)]),
        ("derivative_coefficients", "Return derivative coefficients for a polynomial coefficient list.", "", [([3, 1, 2], None, [1, 4]), ([5], None, []), ([0, 0, 4], None, [0, 8])]),
        ("tribonacci_sequence", "Return the nth item of a three-term Fibonacci-like sequence.", "", [(0, None, 0), (3, None, 1), (6, None, 7)]),
        ("fibonacci_loop_private", "Return the nth value of a two-state recurrence with starting values zero and one.", "", [(0, None, 0), (1, None, 1), (2, None, 1), (8, None, 21)]),
        ("lucas_loop_private", "Return the nth value of a recurrence with starting values two and one.", "", [(0, None, 2), (1, None, 1), (2, None, 3), (6, None, 18)]),
        ("shifted_recurrence_private", "Return the nth recurrence value where each new term is the previous two terms plus one.", "", [(0, None, 0), (1, None, 1), (2, None, 2), (5, None, 12)]),
        ("nested_recurrence_private", "Return a recurrence built by applying a Fibonacci-like update twice per step.", "", [(0, None, 0), (1, None, 1), (2, None, 3), (3, None, 8)]),
        ("rotate_sequence", "Circularly shift a sequence to the right by a non-negative count.", "", [("abcd", 1, "dabc"), ("abcd", 2, "cdab"), ([1, 2, 3], 1, [3, 1, 2])]),
        ("circular_digit_shift", "Circularly shift the digits of an integer to the right; if the shift is larger than the digit count, return the digits reversed.", "", [(100, 2, "001"), (12, 2, "12"), (97, 8, "79"), (12, 1, "21")]),
        ("digit_rotate_right_private", "Rotate digit text to the right by a count while preserving leading zeros.", "", [("100", 2, "001"), ("12", 1, "21"), ("9876", 2, "7698"), ("07", 1, "70")]),
        ("signed_digit_rotate_private", "Rotate the absolute digits of a signed integer to the left and keep the sign.", "", [(-1234, 1, "-2341"), (1234, 2, "3412"), (-9, 5, "-9")]),
        ("multi_step_digit_shift_private", "Apply a circular digit shift repeatedly and return the final digit string.", "", [("1234", (1, 2), "3412"), ("100", (2, 1), "001"), ("7", (5, 3), "7")]),
        ("final_y_vowel_private", "Count vowels in text; a y counts only when it is the final alphabetic character.", "", [("abcde", None, 2), ("sky", None, 1), ("yellow!", None, 2), ("Y", None, 1)]),
        ("suffix_y_vowel_private", "Count vowels after lowercasing; y counts only when the word ends with ly.", "", [("fly", None, 1), ("by", None, 0), ("really", None, 3), ("", None, 0)]),
        ("case_punct_vowel_private", "Count vowels in alphabetic characters only, ignoring punctuation and case.", "", [("A,E-I!", None, 3), ("rhythm", None, 0), ("", None, 0), ("Queue", None, 4)]),
        ("digit_sum_casefold", "Return the sum of decimal digits inside uppercase letters only.", "", [("A1b2C3", None, 4), ("abc123", None, 0), ("Z9", None, 9)]),
        ("uppercase_ascii_sum", "Return the sum of ASCII codes for uppercase letters in a string.", "", [("AzBY", None, 220), ("lower", None, 0), ("Q9R", None, 163)]),
        ("fruit_distribution_private", "Parse a short fruit-count sentence and return the remaining count.", "", [("5 apples and 2 oranges", None, 3), ("10 mangoes and 4 pears", None, 6), ("1 apple and 1 orange", None, 0)]),
        ("pluck_smallest_even", "Return the smallest even item and its index from a list, or an empty list if none exists.", "", [([9, 6, 4, 4], None, [4, 2]), ([1, 3, 5], None, []), ([8, 2, 2], None, [2, 1])]),
        ("frequency_at_least_value", "Return the greatest positive integer whose frequency is at least its own value, or -1 if none exists.", "", [([2, 2, 3, 3, 3, 1], None, 3), ([4, 4, 4], None, -1), ([1, 5, 5, 5, 5, 5], None, 5)]),
        ("alternating_min_max_sort", "Return values by repeatedly taking the current minimum then current maximum.", "", [([4, 1, 3, 2], None, [1, 4, 2, 3]), ([7, 7, 7], None, [7, 7, 7]), ([], None, [])]),
        ("palindrome_list_weight", "Return whether a list is palindromic and its sum does not exceed a limit.", "", [([2, 1, 2], 6, True), ([2, 3], 10, False), ([4, 1, 4], 5, False)]),
        ("smallest_palindrome_changes", "Return the number of mirrored positions that differ in a list.", "", [([1, 2, 1], None, 0), ([1, 2, 3, 4], None, 2), ([5, 6, 7, 6], None, 2)]),
        ("total_match_lengths", "Return the list of strings with the smaller total character count, choosing the first on ties.", "", [(["aa", "b"], ["c"], ["c"]), (["x"], ["y"], ["x"]), (["hi"], ["a", "bc", "d"], ["hi"])]),
        ("multiply_three_primes", "Return whether an integer is the product of exactly three prime factors counted with multiplicity.", "", [(30, None, True), (8, None, True), (12, None, True), (7, None, False), (16, None, False)]),
        ("simple_power", "Return whether one integer is an exact non-negative power of another.", "", [(27, 3, True), (10, 3, False), (1, 5, True), (4, 1, False)]),
        ("cube_number", "Return whether an integer is a perfect cube, including negative cubes.", "", [(27, None, True), (-8, None, True), (45, None, False), (0, None, True)]),
        ("hex_prime_count", "Count hexadecimal characters whose values are prime digits.", "", [("2D0F", None, 2), ("ACE", None, 0), ("357B", None, 4)]),
        ("woodall_number_check", "Return whether n is a Woodall-style number of the form k * 2**k - 1.", "", [(1, None, True), (7, None, True), (383, None, True), (10, None, False)]),
        ("polygonal_octagonal_number", "Return the nth octagonal number for a positive integer n.", "", [(1, None, 1), (2, None, 8), (5, None, 65)]),
        ("polygonal_tetrahedral_number", "Return the nth tetrahedral number.", "", [(1, None, 1), (3, None, 10), (5, None, 35)]),
        ("polygonal_centered_hexagonal_number", "Return the nth centered hexagonal number.", "", [(1, None, 1), (2, None, 7), (4, None, 37)]),
        ("sphere_volume", "Return the volume of a sphere from its radius.", "", [(1, None, 4.1887902047863905), (3, None, 113.09733552923254), (0, None, 0.0)]),
        ("sphere_surface_area", "Return the surface area of a sphere from its radius.", "", [(1, None, 12.566370614359172), (2, None, 50.26548245743669), (0, None, 0.0)]),
        ("sort_by_second", "Sort a list of tuples by the second item.", "sorted(data, key=lambda item: item[1])", [([(3, 2), (1, 1)], None, [(1, 1), (3, 2)]), ([], None, []), ([("x", 3), ("y", 1)], None, [("y", 1), ("x", 3)])]),
        ("nested_flat_sum", "Flatten one or more nested list levels and return the numeric sum.", "", [([1, [2, 3]], None, 6), ([[1], [2, [3]]], None, 6), ([], None, 0)]),
        ("positive_count", "Count positive numbers in a numeric list.", "sum(1 for item in data if item > 0)", [([1, -2, 3], None, 2), ([-1, 0], None, 0), ([5], None, 1)]),
        ("positive_filter", "Return only positive numbers from a numeric list.", "", [([-1, 2, -4, 5, 0], None, [2, 5]), ([], None, []), ([3, -1], None, [3])]),
        ("sublist_contains", "Return whether a list contains a target sublist contiguously.", "", [([1, 2, 3, 4], [2, 3], True), ([1, 2], [2, 1], False), ([1], [], True)]),
        ("equal_tuple_lengths", "Return whether every tuple in a collection has the same length.", "", [([(1, 2), (3, 4)], None, True), ([(1,), (2, 3)], None, False), ([], None, True)]),
        ("sort_list", "Return a sorted copy of a list.", "sorted(data)", [([3, 1, 2], None, [1, 2, 3]), ([], None, []), (["b", "a"], None, ["a", "b"])]),
        ("difference_of_squares_check", "Return whether a non-negative integer can be represented as a difference of two squares.", "", [(0, None, True), (2, None, False), (15, None, True), (20, None, True)]),
        ("same_pattern_sequence", "Return whether values follow the same equality pattern as the pattern sequence.", "", [(["a", "b", "a"], [1, 2, 1], True), (["a", "b", "b"], [1, 2, 1], False), ([], [], True)]),
        ("tuple_all_divisible", "Return tuples whose items are all divisible by a given divisor.", "", [([(2, 4), (3, 6)], 2, [(2, 4)]), ([], 3, []), ([(5, 10)], 5, [(5, 10)])]),
        ("odd_length_check", "Return whether a word or sequence has odd length.", "", [("abc", None, True), ("abcd", None, False), ("", None, False)]),
        ("ascii_mod_char", "Return the lowercase character represented by the sum of character ordinals modulo 26.", "", [("abc", None, "i"), ("A", None, "n"), ("", None, "a")]),
        ("dict_merge_three", "Merge three dictionaries into a new dictionary.", "", [({"__args__": ({"a": 1}, {"b": 2}, {"c": 3})}, None, {"a": 1, "b": 2, "c": 3}), ({"__args__": ({}, {"x": 1}, {})}, None, {"x": 1})]),
        ("frequency_dict", "Return a dictionary mapping each list item to its frequency.", "", [([1, 2, 1], None, {1: 2, 2: 1}), ([], None, {}), (["a", "a"], None, {"a": 2})]),
        ("closest_smaller_number", "Return the closest smaller integer below n.", "", [(10, None, 9), (1, None, 0), (-2, None, -3)]),
        ("longest_word_length", "Return the length of the longest whitespace-separated word.", "", [("hi there", None, 5), ("", None, 0), ("a abc ab", None, 3)]),
        ("substring_in_list", "Return whether a substring is present in any string from a list.", "", [(["abc", "def"], "bc", True), (["abc"], "xy", False), ([], "a", False)]),
        ("overlapping_substring_count", "Count overlapping occurrences of a substring.", "", [("aaaa", "aa", 3), ("abc", "x", 0), ("aaa", "a", 3)]),
        ("spelled_number_sort", "Sort space-delimited number words from zero to nine.", "", [("three one two", None, "one two three"), ("", None, ""), ("nine zero", None, "zero nine")]),
        ("closest_pair_sorted", "Return the closest pair of numbers in ascending order.", "", [([1.0, 2.0, 2.2], None, (2.0, 2.2)), ([5, 1, 3], None, (1, 3)), ([9, 8], None, (8, 9))]),
        ("unique_once_stable", "Return items that occur exactly once while preserving order.", "", [([1, 2, 3, 2, 4], None, [1, 3, 4]), ([1, 1], None, []), ([], None, [])]),
        ("flip_case", "Swap uppercase and lowercase characters in a string.", "data.swapcase()", [("Hello", None, "hELLO"), ("", None, ""), ("AbC", None, "aBc")]),
        ("concat_strings", "Concatenate a list of strings.", "''.join(data)", [(["a", "b"], None, "ab"), ([], None, ""), (["x"], None, "x")]),
        ("filter_by_prefix", "Return strings that start with a provided prefix.", "", [(["abc", "bcd", "array"], "a", ["abc", "array"]), ([], "a", []), (["x"], "z", [])]),
        ("sort_indices_multiple_three", "Sort values at indices divisible by three while preserving other positions.", "", [([5, 9, 1, 8, 3, 7, 4], None, [4, 9, 1, 5, 3, 7, 8]), ([3, 2, 1], None, [3, 2, 1]), ([], None, [])]),
        ("car_race_collision_count", "Return the number of pairwise collisions for n cars moving each direction.", "data * data", [(0, None, 0), (2, None, 4), (5, None, 25)]),
        ("digit_substring_length_sum_count", "Count digit substrings whose digit sum equals their length.", "", [("112112", None, 6), ("111", None, 6), ("12", None, 1)]),
        ("bell_number_sequence", "Return the nth Bell number using a small dynamic-programming triangle.", "", [(0, None, 1), (1, None, 1), (3, None, 5), (5, None, 52)]),
        ("newman_conway_sequence", "Return the nth Newman-Conway-style recurrence value with first two values equal to one.", "", [(1, None, 1), (2, None, 1), (5, None, 3), (8, None, 4)]),
    ]
    transfer_focus_categories = {
        "balanced_brackets_simple",
        "monotonic_sequence",
        "common_elements",
        "largest_prime_factor",
        "arithmetic_series_sum",
        "derivative_coefficients",
        "largest_divisor",
        "prime_factors",
        "divisible_by_11",
        "rescale_to_unit",
        "decode_cyclic",
        "prime_fib_sequence",
        "polynomial_zero_bisection",
        "tribonacci_sequence",
        "fibonacci_loop_private",
        "lucas_loop_private",
        "shifted_recurrence_private",
        "nested_recurrence_private",
        "rotate_sequence",
        "circular_digit_shift",
        "digit_rotate_right_private",
        "signed_digit_rotate_private",
        "multi_step_digit_shift_private",
        "final_y_vowel_private",
        "suffix_y_vowel_private",
        "case_punct_vowel_private",
        "digit_sum_casefold",
        "uppercase_ascii_sum",
        "fruit_distribution_private",
        "pluck_smallest_even",
        "frequency_at_least_value",
        "alternating_min_max_sort",
        "palindrome_list_weight",
        "smallest_palindrome_changes",
        "total_match_lengths",
        "multiply_three_primes",
        "simple_power",
        "cube_number",
        "hex_prime_count",
        "parse_ints",
        "min_three",
        "cube_volume",
        "cube_lateral_surface_area",
        "cylinder_lateral_surface_area",
        "string_char_count",
        "swap_pair",
        "string_odd_index_remove",
        "replace_whitespace",
        "stable_negative_partition",
        "top_k_largest",
        "tuple_frequency_dict",
        "tuple_item_count",
        "count_integer_items",
        "split_list_at_index",
        "tuple_elementwise_division",
        "tuple_elementwise_max",
        "tuple_nested_elementwise_max",
        "insert_before_each",
        "count_primes_below",
        "next_perfect_square",
        "harmonic_sum",
        "list_chunks_every_n",
        "combinations_with_replacement",
        "nonempty_substring_count",
        "median_list",
        "modular_power_two",
        "caesar_decode_shift5",
        "remove_vowels",
        "below_threshold",
        "add_numbers",
        "same_chars",
        "count_vowels",
    }
    semantic_target_categories = [
        "palindrome",
        "caesar_decode_shift5",
        "below_threshold",
        "add_numbers",
        "same_chars",
        "gcd_pair",
        "is_prime",
        "is_anagram",
        "base_digits",
        "triangle_area_sides",
        "uppercase_ascii_sum",
        "pluck_smallest_even",
        "frequency_at_least_value",
        "alternating_min_max_sort",
        "palindrome_list_weight",
        "smallest_palindrome_changes",
        "total_match_lengths",
        "multiply_three_primes",
        "simple_power",
        "cube_number",
        "hex_prime_count",
        "min_three",
        "cube_volume",
        "cube_lateral_surface_area",
        "cylinder_lateral_surface_area",
        "string_char_count",
        "swap_pair",
        "string_odd_index_remove",
        "replace_whitespace",
        "stable_negative_partition",
        "top_k_largest",
        "tuple_frequency_dict",
        "tuple_item_count",
        "count_integer_items",
        "split_list_at_index",
        "tuple_elementwise_division",
        "tuple_elementwise_max",
        "tuple_nested_elementwise_max",
        "insert_before_each",
        "count_primes_below",
        "next_perfect_square",
        "list_chunks_every_n",
        "combinations_with_replacement",
        "nonempty_substring_count",
    ]
    base_category_rows = {row[0]: row for row in categories}
    transfer_focus = [row for row in categories if row[0] in transfer_focus_categories]
    categories = categories + transfer_focus + transfer_focus + transfer_focus
    rows: list[dict[str, Any]] = []
    category_seen: dict[str, int] = {}
    for idx in range(count):
        category, description, expr, tests = categories[(idx + seed) % len(categories)]
        seen = category_seen.get(category, 0)
        split = "eval" if seen % 5 == 0 else "train"
        category_seen[category] = seen + 1
        entry = f"private_{category}_{idx:04d}"
        prompt = f"Write a Python function named {entry}. {description}"
        rows.append(
            {
                "task_id": f"private_code_lm_{category}_{idx:04d}",
                "source_task_id": f"private_{idx:04d}",
                "card_id": "private_code_lm_curriculum",
                "source_id": "local_generated_private_code_curriculum",
                "split": split,
                "category": category,
                "prompt": prompt,
                "entry_point": entry,
                "solution_expr": expr,
                "solution_body": solution_body_for(category, expr),
                "tests": tests_for(entry, tests),
                "tags": [category, "private_code_curriculum"],
                "benchmark_evidence_level": "private_generated_training_or_eval",
                "public_benchmark": False,
                "license_spdx": "local-generated-provenance-only",
                "candidate_expression_eligible": bool(expr.strip()),
            }
        )
    target_rows_added = 0
    for repeat in range(8):
        for category in semantic_target_categories:
            if category not in base_category_rows:
                continue
            category, description, expr, tests = base_category_rows[category]
            idx = count + target_rows_added
            target_rows_added += 1
            entry = f"private_semantic_{category}_{idx:04d}"
            prompt = (
                f"Write a Python function named {entry}. "
                f"Private generated semantic-transfer task: {description}"
            )
            rows.append(
                {
                    "task_id": f"private_code_lm_semantic_target_{category}_{idx:04d}",
                    "source_task_id": f"private_semantic_target_{idx:04d}",
                    "card_id": "private_code_lm_semantic_target_curriculum",
                    "source_id": "local_generated_private_semantic_target_curriculum",
                    "split": "train",
                    "category": category,
                    "prompt": prompt,
                    "entry_point": entry,
                    "solution_expr": expr,
                    "solution_body": solution_body_for(category, expr),
                    "tests": tests_for(entry, tests),
                    "tags": [category, "private_semantic_target_curriculum"],
                    "benchmark_evidence_level": "private_generated_training_or_eval",
                    "public_benchmark": False,
                    "license_spdx": "local-generated-provenance-only",
                    "candidate_expression_eligible": bool(expr.strip()),
                    "provenance": {
                        "policy": "project_theseus_private_semantic_target_curriculum_v1",
                        "public_benchmark_answers_used": False,
                        "public_tests_used": False,
                    },
                }
            )
    return rows


def load_extra_private_train(path: Path, *, max_rows: int) -> list[dict[str, Any]]:
    if max_rows <= 0 or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in read_jsonl(path):
        task_id = safe_name(raw.get("task_id") or "")
        prompt = str(raw.get("prompt") or "").strip()
        entry_point = safe_name(raw.get("entry_point") or "open_code_func")
        solution_expr = str(raw.get("solution_expr") or "").strip()
        solution_body = str(raw.get("solution_body") or "").strip()
        if not task_id or not prompt or not entry_point or (not solution_expr and not solution_body):
            continue
        if task_id in seen:
            continue
        license_spdx = str(raw.get("license_spdx") or "")
        if not license_spdx or license_spdx.lower() in {"unknown", "noassertion"}:
            continue
        emitted_task_id = task_id if task_id.startswith("extra_") else f"extra_{task_id}"
        rows.append(
            {
                "task_id": emitted_task_id,
                "source_task_id": str(raw.get("source_task_id") or task_id),
                "card_id": str(raw.get("card_id") or "open_code_permissive_pantry"),
                "source_id": str(raw.get("source_id") or "open_code_permissive_pantry"),
                "split": "train",
                "category": str(raw.get("category") or "open_code_expr"),
                "prompt": prompt,
                "entry_point": entry_point,
                "solution_expr": solution_expr or first_return_expression(solution_body),
                "solution_body": solution_body or body_from_expr(solution_expr),
                "tests": "",
                "tags": [str(tag) for tag in raw.get("tags", [])] if isinstance(raw.get("tags"), list) else ["open_code_permissive_pantry"],
                "benchmark_evidence_level": str(raw.get("benchmark_evidence_level") or "permissive_open_source_train_only"),
                "public_benchmark": False,
                "license_spdx": license_spdx,
                "candidate_expression_eligible": bool(raw.get("candidate_expression_eligible", False)),
                "provenance": raw.get("provenance") if isinstance(raw.get("provenance"), dict) else {},
                "decoder_contract": raw.get("decoder_contract") if isinstance(raw.get("decoder_contract"), dict) else {},
            }
        )
        seen.add(task_id)
        if len(rows) >= max_rows:
            break
    return rows


def split_path_list(raw: str) -> list[str]:
    out: list[str] = []
    for chunk in str(raw or "").replace(",", ";").split(";"):
        path = chunk.strip()
        if path:
            out.append(path)
    return out


def load_extra_private_train_many(raw_paths: str, *, max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    paths = split_path_list(raw_paths)
    if max_rows <= 0 or not paths:
        return rows

    # High-transfer curriculum files represent different donor concepts. A
    # sequential cap quietly over-samples whichever file appears first, so a
    # 2k-row budget can become "type contracts only" and starve edge,
    # admissibility, algorithmic planning, and execution-shaped rows. Round
    # robin keeps capped runs broad while preserving deterministic ordering
    # within each source file.
    buckets: list[tuple[str, list[dict[str, Any]]]] = []
    for raw_path in paths:
        normalized_path = str(Path(raw_path)).replace("\\", "/").lower()
        if normalized_path in DIAGNOSTIC_ONLY_PRIVATE_ROW_PATHS:
            continue
        resolved = resolve(raw_path)
        source_rows = load_extra_private_train(resolved, max_rows=max_rows)
        if source_rows:
            buckets.append((rel(resolved), source_rows))

    indices = [0 for _ in buckets]
    while len(rows) < max_rows:
        advanced = False
        for bucket_index, (source_label, source_rows) in enumerate(buckets):
            while indices[bucket_index] < len(source_rows):
                row = source_rows[indices[bucket_index]]
                indices[bucket_index] += 1
                task_id = str(row.get("task_id") or "")
                if not task_id or task_id in seen:
                    continue
                row_concept = row.get("residual_concept") or row.get("category") or row.get("concept") or ""
                if is_non_promotable_diagnostic_concept(row_concept):
                    continue
                row = dict(row)
                row["high_transfer_source_jsonl"] = source_label
                rows.append(row)
                seen.add(task_id)
                advanced = True
                break
            if len(rows) >= max_rows:
                break
        if not advanced:
            break
    return rows


def body_from_expr(expr: str) -> str:
    return f"return {expr.strip()}"


def first_return_expression(body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("return "):
            return stripped[len("return ") :].strip()
    return ""


def solution_body_for(category: str, expr: str) -> str:
    bodies = {
        "opposite_signs": "if data == 0 or other == 0:\n    return False\nreturn (data < 0) != (other < 0)",
        "all_prefixes": "out = []\nfor idx in range(1, len(data) + 1):\n    out.append(data[:idx])\nreturn out",
        "string_sequence": "parts = []\nfor idx in range(data + 1):\n    parts.append(str(idx))\nreturn ' '.join(parts)",
        "largest_concat": "items = sorted((str(item) for item in data), key=lambda item: item * 8, reverse=True)\nreturn int(''.join(items))",
        "count_vowels": "text = data.lower()\ntotal = 0\nfor idx, ch in enumerate(text):\n    if ch in 'aeiou' or (ch == 'y' and idx == len(text) - 1):\n        total += 1\nreturn total",
        "one_less_than_twice_reverse": "reversed_digits = int(str(abs(data))[::-1])\nreturn data == 2 * reversed_digits - 1",
        "median_list": "items = sorted(data)\nmid = len(items) // 2\nif len(items) % 2 == 1:\n    return items[mid]\nreturn (items[mid - 1] + items[mid]) / 2",
        "modular_power_two": "result = 1\nfor _ in range(data):\n    result = (result * 2) % other\nreturn result",
        "caesar_decode_shift5": "out = []\nfor ch in data:\n    out.append(chr(((ord(ch) - 5 - ord('a')) % 26) + ord('a')))\nreturn ''.join(out)",
        "remove_vowels": "out = []\nfor ch in data:\n    if ch.lower() not in 'aeiou':\n        out.append(ch)\nreturn ''.join(out)",
        "below_threshold": "for item in data:\n    if item >= other:\n        return False\nreturn True",
        "add_numbers": "return data + other",
        "same_chars": "return set(data) == set(other)",
        "filter_integers": "out = []\nfor item in data:\n    if isinstance(item, int):\n        out.append(item)\nreturn out",
        "replace_whitespace": "return data.replace(' ', other)",
        "stable_negative_partition": "head = list(data[:other])\ntail = list(data[other:])\nnegatives = []\nnonnegatives = []\nfor item in head:\n    if item < 0:\n        negatives.append(item)\n    else:\n        nonnegatives.append(item)\nreturn negatives + nonnegatives + tail",
        "top_k_largest": "items = sorted(data, reverse=True)\nreturn items[:other]",
        "cube_volume": "return data * data * data",
        "cube_lateral_surface_area": "return 4 * data * data",
        "cylinder_lateral_surface_area": "return 2 * 3.141592653589793 * data * other",
        "list_tail_replace": "return data[:-1] + list(other)",
        "tuple_frequency_dict": "counts = {}\nfor item in data:\n    counts[item] = counts.get(item, 0) + 1\nreturn counts",
        "tuple_item_count": "return data.count(other)",
        "count_primes_below": "total = 0\nfor value in range(2, data):\n    is_prime_value = True\n    for divisor in range(2, int(value ** 0.5) + 1):\n        if value % divisor == 0:\n            is_prime_value = False\n            break\n    if is_prime_value:\n        total += 1\nreturn total",
        "next_perfect_square": "root = 0\nwhile root * root <= data:\n    root += 1\nreturn root * root",
        "insert_before_each": "out = []\nfor item in data:\n    out.append(other)\n    out.append(item)\nreturn out",
        "tuple_nested_elementwise_max": "out = []\nfor left_pair, right_pair in zip(data, other):\n    out.append(tuple(max(left, right) for left, right in zip(left_pair, right_pair)))\nreturn tuple(out)",
        "list_chunks_every_n": "return [data[idx:idx + other] for idx in range(0, len(data), other)]",
        "combinations_with_replacement": "out = []\ndef build(start, current):\n    if len(current) == other:\n        out.append(tuple(current))\n        return\n    for idx in range(start, len(data)):\n        build(idx, current + [data[idx]])\nbuild(0, [])\nreturn out",
        "sort_by_second": "out = list(data)\nfor i in range(len(out)):\n    for j in range(i + 1, len(out)):\n        if out[j][1] < out[i][1]:\n            out[i], out[j] = out[j], out[i]\nreturn out",
        "positive_count": "total = 0\nfor item in data:\n    if item > 0:\n        total += 1\nreturn total",
        "negative_count": "total = 0\nfor item in data:\n    if item < 0:\n        total += 1\nreturn total",
        "safe_head": "if data:\n    return data[0]\nreturn other",
        "dict_required_keys": "for key in other:\n    if key not in data:\n        return False\nreturn True",
        "distinct_count": "if isinstance(data, str):\n    return len(set(data.lower()))\nreturn len(set(data))",
        "stable_dedupe": "out = []\nseen = set()\nfor item in data:\n    if item not in seen:\n        seen.add(item)\n        out.append(item)\nreturn out",
        "largest_divisor": "best = 1\nfor item in range(1, data):\n    if data % item == 0:\n        best = item\nreturn best",
        "factors": "out = []\nfor item in range(1, data + 1):\n    if data % item == 0:\n        out.append(item)\nreturn out",
        "prime_factors": "out = []\nvalue = data\nfactor = 2\nwhile factor * factor <= value:\n    while value % factor == 0:\n        out.append(factor)\n        value //= factor\n    factor += 1\nif value > 1:\n    out.append(value)\nreturn out",
        "divisible_by_11": "return data % 11 == 0",
        "nested_sum": "total = 0\nfor item in data:\n    if isinstance(item, list):\n        total += sum(item)\n    else:\n        total += item\nreturn total",
        "rescale_to_unit": "low = min(data)\nhigh = max(data)\nreturn [(item - low) / (high - low) for item in data]",
        "decode_cyclic": "out = []\nfor idx in range(0, len(data), 3):\n    group = data[idx:idx + 3]\n    if len(group) == 3:\n        out.append(group[-1] + group[:-1])\n    else:\n        out.append(group)\nreturn ''.join(out)",
        "prime_fib_sequence": "found = 0\na = 1\nb = 1\nwhile True:\n    a, b = b, a + b\n    is_prime_value = a > 1\n    for divisor in range(2, int(a ** 0.5) + 1):\n        if a % divisor == 0:\n            is_prime_value = False\n            break\n    if is_prime_value:\n        found += 1\n        if found == data:\n            return a",
        "polynomial_zero_bisection": "left = -1.0\nright = 1.0\ndef value_at(x):\n    total = 0.0\n    power = 1.0\n    for coeff in data:\n        total += coeff * power\n        power *= x\n    return total\nwhile value_at(left) * value_at(right) > 0:\n    left *= 2\n    right *= 2\nfor _ in range(60):\n    mid = (left + right) / 2\n    if value_at(left) * value_at(mid) <= 0:\n        right = mid\n    else:\n        left = mid\nreturn (left + right) / 2",
        "closest_pair": "best = None\nbest_dist = None\nfor idx, left in enumerate(data):\n    for right in data[idx + 1:]:\n        dist = abs(left - right)\n        if best_dist is None or dist < best_dist:\n            best_dist = dist\n            best = (left, right)\nreturn best",
        "sum_squares": "total = 0\nfor item in data:\n    total += item * item\nreturn total",
        "average_or_zero": "if not data:\n    return 0\nreturn sum(data) / len(data)",
        "gcd_pair": "left = abs(data)\nright = abs(other)\nbest = 1\nfor item in range(1, min(left, right) + 1):\n    if left % item == 0 and right % item == 0:\n        best = item\nreturn best",
        "is_prime": "if data <= 1:\n    return False\nfor item in range(2, int(data ** 0.5) + 1):\n    if data % item == 0:\n        return False\nreturn True",
        "powers_of_two": "out = []\nvalue = 1\nfor _ in range(data):\n    out.append(value)\n    value *= 2\nreturn out",
        "flatten_once": "out = []\nfor item in data:\n    if isinstance(item, list):\n        out.extend(item)\n    else:\n        out.append(item)\nreturn out",
        "parse_ints": "out = []\nfor part in data.split():\n    if part.lstrip('-').isdigit():\n        out.append(int(part))\nreturn out",
        "symbol_beat_parser": "beats = {'o': 4, 'o|': 2, '.|': 1}\nout = []\nfor part in data.split():\n    out.append(beats[part])\nreturn out",
        "remove_none": "out = []\nfor item in data:\n    if item is not None:\n        out.append(item)\nreturn out",
        "index_or_minus_one": "if other in data:\n    return data.index(other)\nreturn -1",
        "count_truthy": "total = 0\nfor item in data:\n    if item:\n        total += 1\nreturn total",
        "matrix_diagonal": "out = []\nlimit = min(len(data), len(data[0]) if data else 0)\nfor idx in range(limit):\n    out.append(data[idx][idx])\nreturn out",
        "extract_def_name": "for line in data.splitlines():\n    stripped = line.strip()\n    if stripped.startswith('def ') and '(' in stripped:\n        return stripped[4:].split('(', 1)[0].strip()\nreturn ''",
        "sorted_unique_values": "return sorted(set(data))",
        "common_elements": "return sorted(set(data) & set(other))",
        "sort_even_index_values": "out = list(data)\nevens = sorted(out[::2])\nslot = 0\nfor idx in range(0, len(out), 2):\n    out[idx] = evens[slot]\n    slot += 1\nreturn out",
        "increment_each_item": "out = []\nfor item in data:\n    out.append(item + 1)\nreturn out",
        "count_digit_under_divisibility": "total = 0\nfor item in range(data):\n    if item % other[0] == 0 or item % other[1] == 0:\n        total += str(item).count(str(other[2]))\nreturn total",
        "two_sum_zero_exists": "seen = set()\nfor item in data:\n    if -item in seen:\n        return True\n    seen.add(item)\nreturn False",
        "three_sum_zero_exists": "n = len(data)\nfor i in range(n):\n    for j in range(i + 1, n):\n        for k in range(j + 1, n):\n            if data[i] + data[j] + data[k] == 0:\n                return True\nreturn False",
        "base_digits": "if data == 0:\n    return '0'\ndigits = ''\nwhile data > 0:\n    digits = str(data % other) + digits\n    data //= other\nreturn digits",
        "triangle_area_product": "return data * other / 2",
        "triangle_area_sides": "a = data\nb = other\nc = extra[0] if extra else 0\nif a + b <= c or a + c <= b or b + c <= a:\n    return -1\ns = (a + b + c) / 2\nreturn round((s * (s - a) * (s - b) * (s - c)) ** 0.5, 2)",
        "balanced_brackets_simple": "pairs = {')': '(', ']': '[', '}': '{', '>': '<'}\nstack = []\nfor ch in data:\n    if ch in '([{<':\n        stack.append(ch)\n    elif ch in pairs:\n        if not stack or stack[-1] != pairs[ch]:\n            return False\n        stack.pop()\nreturn not stack",
        "monotonic_sequence": "if len(data) < 2:\n    return True\nnondecreasing = True\nnonincreasing = True\nfor idx in range(1, len(data)):\n    if data[idx] < data[idx - 1]:\n        nondecreasing = False\n    if data[idx] > data[idx - 1]:\n        nonincreasing = False\nreturn nondecreasing or nonincreasing",
        "largest_prime_factor": "best = 1\nvalue = data\nfactor = 2\nwhile factor * factor <= value:\n    while value % factor == 0:\n        best = factor\n        value //= factor\n    factor += 1\nif value > 1:\n    best = value\nreturn best",
        "arithmetic_series_sum": "total = 0\nfor item in range(data + 1):\n    total += item\nreturn total",
        "derivative_coefficients": "out = []\nfor idx in range(1, len(data)):\n    out.append(idx * data[idx])\nreturn out",
        "tribonacci_sequence": "values = [0, 0, 1]\nif data < len(values):\n    return values[data]\nfor _ in range(3, data + 1):\n    values.append(values[-1] + values[-2] + values[-3])\nreturn values[data]",
        "fibonacci_loop_private": "a = 0\nb = 1\nif data == 0:\n    return a\nfor _ in range(1, data):\n    a, b = b, a + b\nreturn b",
        "lucas_loop_private": "a = 2\nb = 1\nif data == 0:\n    return a\nfor _ in range(1, data):\n    a, b = b, a + b\nreturn b",
        "shifted_recurrence_private": "a = 0\nb = 1\nif data == 0:\n    return a\nfor _ in range(1, data):\n    a, b = b, a + b + 1\nreturn b",
        "nested_recurrence_private": "a = 0\nb = 1\nfor _ in range(data):\n    a, b = b, a + b\n    a, b = b, a + b\nreturn a",
        "rotate_sequence": "if not data:\n    return data\nshift = other % len(data)\nif shift == 0:\n    return data\nreturn data[-shift:] + data[:-shift]",
        "circular_digit_shift": "digits = str(data)\nif other > len(digits):\n    return digits[::-1]\nshift = other % len(digits)\nif shift == 0:\n    return digits\nreturn digits[-shift:] + digits[:-shift]",
        "digit_rotate_right_private": "digits = str(data)\nif not digits:\n    return digits\nshift = other % len(digits)\nif shift == 0:\n    return digits\nreturn digits[-shift:] + digits[:-shift]",
        "signed_digit_rotate_private": "sign = '-' if data < 0 else ''\ndigits = str(abs(data))\nif not digits:\n    return sign + digits\nshift = other % len(digits)\nrotated = digits[shift:] + digits[:shift]\nreturn sign + rotated",
        "multi_step_digit_shift_private": "digits = str(data)\nif not digits:\n    return digits\nfor _ in range(other[1]):\n    shift = other[0] % len(digits)\n    if shift:\n        digits = digits[-shift:] + digits[:-shift]\nreturn digits",
        "final_y_vowel_private": "text = ''.join(ch.lower() for ch in data if ch.isalpha())\ntotal = 0\nfor idx, ch in enumerate(text):\n    if ch in 'aeiou' or (ch == 'y' and idx == len(text) - 1):\n        total += 1\nreturn total",
        "suffix_y_vowel_private": "text = data.strip().lower()\ntotal = 0\nfor idx, ch in enumerate(text):\n    if ch in 'aeiou' or (ch == 'y' and text.endswith('ly') and idx == len(text) - 1):\n        total += 1\nreturn total",
        "case_punct_vowel_private": "total = 0\nfor ch in data.lower():\n    if ch.isalpha() and ch in 'aeiou':\n        total += 1\nreturn total",
        "digit_sum_casefold": "total = 0\nprevious_upper = False\nfor ch in data:\n    if ch.isupper():\n        previous_upper = True\n    elif ch.isdigit() and previous_upper:\n        total += int(ch)\n        previous_upper = False\n    else:\n        previous_upper = False\nreturn total",
        "uppercase_ascii_sum": "total = 0\nfor ch in data:\n    if ch.isupper():\n        total += ord(ch)\nreturn total",
        "fruit_distribution_private": "numbers = []\nfor part in data.replace(',', ' ').split():\n    if part.isdigit():\n        numbers.append(int(part))\nif len(numbers) < 2:\n    return 0\nreturn numbers[0] - numbers[1]",
        "pluck_smallest_even": "best = None\nbest_idx = -1\nfor idx, item in enumerate(data):\n    if item % 2 == 0 and (best is None or item < best):\n        best = item\n        best_idx = idx\nif best is None:\n    return []\nreturn [best, best_idx]",
        "frequency_at_least_value": "counts = {}\nfor item in data:\n    counts[item] = counts.get(item, 0) + 1\nbest = -1\nfor item, count in counts.items():\n    if item > 0 and count >= item and item > best:\n        best = item\nreturn best",
        "alternating_min_max_sort": "items = sorted(data)\nout = []\ntake_low = True\nwhile items:\n    if take_low:\n        out.append(items.pop(0))\n    else:\n        out.append(items.pop())\n    take_low = not take_low\nreturn out",
        "palindrome_list_weight": "if data != data[::-1]:\n    return False\nreturn sum(data) <= other",
        "smallest_palindrome_changes": "total = 0\nfor idx in range(len(data) // 2):\n    if data[idx] != data[-idx - 1]:\n        total += 1\nreturn total",
        "total_match_lengths": "left = sum(len(item) for item in data)\nright = sum(len(item) for item in other)\nif left <= right:\n    return data\nreturn other",
        "multiply_three_primes": "value = data\ncount = 0\nfactor = 2\nwhile factor <= value:\n    while value % factor == 0:\n        count += 1\n        value //= factor\n    factor += 1\nreturn count == 3",
        "simple_power": "if data == 1:\n    return True\nif other <= 1:\n    return False\nvalue = 1\nwhile value < data:\n    value *= other\nreturn value == data",
        "cube_number": "value = abs(data)\nroot = 0\nwhile root * root * root < value:\n    root += 1\nreturn root * root * root == value",
        "hex_prime_count": "total = 0\nfor ch in data:\n    if ch in '2357BD':\n        total += 1\nreturn total",
        "woodall_number_check": "k = 1\nwhile k * (2 ** k) - 1 <= data:\n    if k * (2 ** k) - 1 == data:\n        return True\n    k += 1\nreturn False",
        "polygonal_octagonal_number": "return data * (3 * data - 2)",
        "polygonal_tetrahedral_number": "return data * (data + 1) * (data + 2) // 6",
        "polygonal_centered_hexagonal_number": "return 3 * data * (data - 1) + 1",
        "sphere_volume": "return 4 / 3 * 3.141592653589793 * data ** 3",
        "sphere_surface_area": "return 4 * 3.141592653589793 * data * data",
        "nested_flat_sum": "total = 0\nstack = list(data)\nwhile stack:\n    item = stack.pop()\n    if isinstance(item, list):\n        stack.extend(item)\n    else:\n        total += item\nreturn total",
        "positive_filter": "out = []\nfor item in data:\n    if item > 0:\n        out.append(item)\nreturn out",
        "sublist_contains": "if other == []:\n    return True\nfor idx in range(0, len(data) - len(other) + 1):\n    if data[idx:idx + len(other)] == other:\n        return True\nreturn False",
        "equal_tuple_lengths": "if not data:\n    return True\nsize = len(data[0])\nfor item in data:\n    if len(item) != size:\n        return False\nreturn True",
        "sort_list": "return sorted(data)",
        "difference_of_squares_check": "return data % 4 != 2",
        "same_pattern_sequence": "if len(data) != len(other):\n    return False\nleft = {}\nright = {}\nfor a, b in zip(data, other):\n    if left.get(a, b) != b or right.get(b, a) != a:\n        return False\n    left[a] = b\n    right[b] = a\nreturn True",
        "tuple_all_divisible": "out = []\nfor item in data:\n    if all(value % other == 0 for value in item):\n        out.append(item)\nreturn out",
        "odd_length_check": "return len(data) % 2 == 1",
        "ascii_mod_char": "total = 0\nfor ch in data:\n    total += ord(ch)\nreturn chr(total % 26 + ord('a'))",
        "dict_merge_three": "out = {}\nout.update(data)\nout.update(other)\nfor item in extra:\n    out.update(item)\nreturn out",
        "frequency_dict": "counts = {}\nfor item in data:\n    counts[item] = counts.get(item, 0) + 1\nreturn counts",
        "closest_smaller_number": "return data - 1",
        "longest_word_length": "words = data.split()\nif not words:\n    return 0\nreturn max(len(word) for word in words)",
        "substring_in_list": "for item in data:\n    if other in item:\n        return True\nreturn False",
        "overlapping_substring_count": "if other == '':\n    return 0\ntotal = 0\nfor idx in range(0, len(data) - len(other) + 1):\n    if data[idx:idx + len(other)] == other:\n        total += 1\nreturn total",
        "spelled_number_sort": "order = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}\nwords = data.split()\nwords.sort(key=lambda word: order[word])\nreturn ' '.join(words)",
        "closest_pair_sorted": "best = None\nbest_dist = None\nitems = sorted(data)\nfor idx in range(len(items) - 1):\n    dist = abs(items[idx + 1] - items[idx])\n    if best_dist is None or dist < best_dist:\n        best_dist = dist\n        best = (items[idx], items[idx + 1])\nreturn best",
        "unique_once_stable": "counts = {}\nfor item in data:\n    counts[item] = counts.get(item, 0) + 1\nout = []\nfor item in data:\n    if counts[item] == 1:\n        out.append(item)\nreturn out",
        "filter_by_prefix": "out = []\nfor item in data:\n    if item.startswith(other):\n        out.append(item)\nreturn out",
        "sort_indices_multiple_three": "out = list(data)\nvalues = sorted(out[::3])\nslot = 0\nfor idx in range(0, len(out), 3):\n    out[idx] = values[slot]\n    slot += 1\nreturn out",
        "digit_substring_length_sum_count": "total = 0\nfor start in range(len(data)):\n    digit_sum = 0\n    for end in range(start, len(data)):\n        digit_sum += int(data[end])\n        if digit_sum == end - start + 1:\n            total += 1\nreturn total",
        "bell_number_sequence": "bell = [[0 for _ in range(data + 1)] for _ in range(data + 1)]\nbell[0][0] = 1\nfor i in range(1, data + 1):\n    bell[i][0] = bell[i - 1][i - 1]\n    for j in range(1, i + 1):\n        bell[i][j] = bell[i - 1][j - 1] + bell[i][j - 1]\nreturn bell[data][0]",
        "newman_conway_sequence": "if data <= 2:\n    return 1\nvalues = [0, 1, 1]\nfor n in range(3, data + 1):\n    previous = values[n - 1]\n    values.append(values[previous] + values[n - previous])\nreturn values[data]",
    }
    return bodies.get(category, body_from_expr(expr))


def tests_for(entry: str, cases: list[tuple[Any, Any, Any]]) -> str:
    lines = []
    for left, right, expected in cases:
        if isinstance(left, dict) and "__args__" in left:
            args = ", ".join(repr(item) for item in left["__args__"])
            lines.append(f"assert {entry}({args}) == {expected!r}")
        elif right is None:
            lines.append(f"assert {entry}({left!r}) == {expected!r}")
        else:
            lines.append(f"assert {entry}({left!r}, {right!r}) == {expected!r}")
    return "\n".join(lines) + "\n"

