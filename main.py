def nested_list(thelist):
    if isinstance(thelist[0], list):
        return (len(thelist), ) + nested_list(thelist[0])
    else:
        return (len(thelist), )

print(nested_list([[1, 2, 3, 4], [1, 1, 1, 1]]))
