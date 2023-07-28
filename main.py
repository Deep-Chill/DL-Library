depth = 0
length = 0


def nested_list(thelist):
    depth = 0
    for i in thelist:
        depth += 1
        if type(i) == list:
            return (len(thelist), nested_list(i))
        else:
            return len(thelist)

print(nested_list([[[[[1, 2, 3, 4], [1, 1, 1, 1]]]]]))