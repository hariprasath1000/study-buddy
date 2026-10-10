from studybuddy import __name__ as package_name


def test_package_is_importable() -> None:
    assert package_name == "studybuddy"
