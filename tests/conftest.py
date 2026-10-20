"""Require explicit real-video selection even when CLI -m overrides pytest defaults."""
import ast


def requests_real_video(expression):
    try:
        tree = ast.parse(expression or "False", mode="eval")
    except SyntaxError:
        return False

    def positive(node, negated=False):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return positive(node.operand, not negated)
        if isinstance(node, ast.Name):
            return node.id == "real_video" and not negated
        return any(positive(child, negated) for child in ast.iter_child_nodes(node))

    return positive(tree)


def pytest_collection_modifyitems(config, items):
    if requests_real_video(config.getoption("markexpr")):
        return
    deselected = [item for item in items if item.get_closest_marker("real_video") is not None]
    if deselected:
        items[:] = [item for item in items if item not in deselected]
        config.hook.pytest_deselected(items=deselected)
