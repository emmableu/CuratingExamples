import itertools

from anytree import Node, RenderTree
import json
import copy

# with open('scratch.json') as j_file:e)
#     json_data = json.load(j_fil

def json2tree(json_file, root_name):
    root = Node(root_name, opcode='targets')
    node_map = {}
    with open(json_file) as json_file:
        data = json.load(json_file)
        targets_obj = data['targets']
        for target_obj in targets_obj:
            opcode = 'stage' if target_obj['isStage'] is True else 'sprite'
            target_node = Node(target_obj['name'], parent=root, opcode=opcode)
            # node_map[target_obj['name']] = target_node
            blocks = target_obj['blocks']
            for key in blocks.keys():
                block = blocks[key]
                tmpBlkNode = Node(key, opcode=block['opcode'], childKey=block['next'], parentKey=block['parent'])
                node_map[key] = tmpBlkNode
                if block['parent'] is None:
                    tmpBlkNode.parent = target_node
            for key in blocks.keys():
                nd = node_map[key]
                if nd.childKey in node_map.keys():
                    nd.children = [node_map[nd.childKey]]
                if nd.parentKey in node_map.keys():
                    nd.parent = node_map[nd.parentKey]
    return root

def json_data2tree(data, root_name):
    root = Node(root_name, opcode='targets')
    node_map = {}
    targets_obj = data['targets']
    for target_obj in targets_obj:
        opcode = 'stage' if target_obj['isStage'] is True else 'sprite'
        target_node = Node(target_obj['name'], parent=root, opcode=opcode)
        # node_map[target_obj['name']] = target_node
        blocks = target_obj['blocks']
        for key in blocks.keys():
            block = blocks[key]
            ck = block.get('next')
            pk = block.get('parent')
            tmpBlkNode = Node(key, opcode=block['opcode'], childKey=ck, parentKey=pk)
            node_map[key] = tmpBlkNode
            if block.get('parent') is None:
                tmpBlkNode.parent = target_node
        for key in blocks.keys():
            nd = node_map[key]
            if nd.childKey in node_map:
                nd.children = [node_map[nd.childKey]]
            if nd.parentKey in node_map:
                nd.parent = node_map[nd.parentKey]
    return root


class PQGram:
    def __init__(self, p, q, node_list):
        self.p = p
        self.q = q
        self.string = "|".join(node_list)
        self.count = 1


class PQGramSet:
    def __init__(self):
        self.set = set()
        self.string_set = set()

    def add_pq_gram(self, new_pq_gram):
        for pg in self.set:
            if pg.string == new_pq_gram.string:
                pg.count += 1
                return
        self.set.add(new_pq_gram)


def dfs_visit_node(pq_gram_set, node, p, q):
    gram_list = ["empty"] * (p + q)
    if not node:
        return
    node_to_parent = copy.deepcopy(node)
    for parent_level in range(1, p):
        if node_to_parent.parent:
            gram_list[p - parent_level - 1] = node_to_parent.parent.opcode
            node_to_parent = node_to_parent.parent.opcode
    gram_list[p - 1] = node.opcode

    if not node.children:
        pq_gram_set.add_pq_gram(PQGram(p, q, gram_list))
        return
    children_list = [child.opcode for child in node.children]
    children_list = ["empty", "empty"] + children_list + ["empty", "empty"]
    for i in range(len(children_list) - 2):
        gram_list[-q:] = children_list[i:i + q]
        new_pq_gram = PQGram(p, q, gram_list)
        pq_gram_set.add_pq_gram(new_pq_gram)

    for child in node.children:
        dfs_visit_node(pq_gram_set, child, p, q)
    return pq_gram_set


def get_code_shape(json_file, root_name, pair_list):
    res_list = {}
    root = json2tree(json_file, root_name)
    for pair in pair_list:
        tmp_set = PQGramSet()
        dfs_visit_node(tmp_set, root, pair[0], pair[1])
        for pq_gram in tmp_set.set:
            res_list[pq_gram.string] = pq_gram.count
    return res_list


def get_code_shape_from_data(data, root_name, pair_list):
    res_list = {}
    root = json_data2tree(data, root_name)
    for pair in pair_list:
        tmp_set = PQGramSet()
        dfs_visit_node(tmp_set, root, pair[0], pair[1])
        for pq_gram in tmp_set.set:
            res_list[pq_gram.string] = pq_gram.count
    return res_list


def get_all_path(res, stack, root):
    if root is None:
        return res
    stack.append(root)
    if not root.children:
        res.append(copy.deepcopy(stack))
    else:
        for child in root.children:
            get_all_path(res, stack, child)
    stack.pop()
    return res


# helper functions for combination
def backtrack(res, li, pa, n, idx):
    if len(li) == n:
        res.append(copy.deepcopy(li))
        return
    i = idx
    while i <= n:
        li.append(pa[i])
        backtrack(res, li, pa, n, i+1)
        li.pop()
        i += 1

 # [[2, 0], [3, 0]]
def combination(data, num_list):
    node_res = {}
    res = {}
    root = json_data2tree(data, 'targets')
    paths = get_all_path([], [], root)
    # print(paths)
    for n in num_list:
        n = n[0]
        for p in paths:
            for c in itertools.combinations(p, n):
                node_str = str(c)
                opcode_str = '|'.join(x.opcode for x in list(c))
                if node_str in node_res.keys():
                    node_res[node_str] += 1
                    res[opcode_str] += 1
                else:
                    node_res[node_str] = 1
                    res[opcode_str] = 1
    return res



def snap_json2tree(json_file):
    json_file = 'example.json'
    with open(json_file) as json_file:
        data = json.load(json_file)
    root = Node(data['type'])
    cur_root = root
    while data['children']:
        for child in data['children']:
            child = Node(child['type'], parent = cur_root )
            cur_root = child
    r = RenderTree(root)
    print(r)
    return root




json_file = 'example.json'
with open(json_file) as json_file:
    data = json.load(json_file)

root = Node(data['type'])
