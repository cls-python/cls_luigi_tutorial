from luigi.task import flatten


def print_tree(task, indent='', last=True):

    name = task.__class__.__name__
    result = '\n' + indent
    if (last):
        result += '└─--'
        indent += '    '
    else:
        result += '|---'
        indent += '|   '
    result += '[{0}]'.format(name)
    children = flatten(task.requires())
    for index, child in enumerate(children):
        result += print_tree(child, indent, (index+1) == len(children))
    return result