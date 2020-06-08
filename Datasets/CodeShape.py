import sys

sys.path.append("/Users/wwang33/Documents/ProgSnap2DataAnalysis/Datasets")
from Dataset import *

sys.path.append("/Users/wwang33/Documents/ProgSnap2DataAnalysis/Models")
import json
import copy
from anytree import Node, RenderTree
from tqdm import tqdm

# This class represents a directed graph using corresponding json code

system_values = ["yPosition", "xPosition", "direction", "random position", "mouse-pointer", "center", "any key", "up arrow",
                 "down arrow", "right arrow", "left arrow", "space", "clicked", "pressed", "dropped", "mouse-entered", "mouse-departed",
                 "scrolled-up", "scrolled-down", "stopped", "any message", "message", "Sprite", "myself", "name", "width", "height",
                 "pixels", "current", "color", "saturation", "brightness", "ghost", "fisheye", "whirl", "pixelate", "mosaic",
                 "negative", "size", "front", "back", "edge", "pen trails", "answer", "space", "Stage", "Sprite", "dangling?", "rotation x",
                 "rotation y", "center x", "center y", "distance", "hue", "saturation", "brightness", "transparency", "r-g-b-a", "sprites",
                 "neighbors", "self", "other sprites", "clones", "other clones", "parts", "anchor", "children", "parent",
                 "temporary?", "name", "costume", "costumes", "sounds", "draggable?", "left", "right", "top", "bottom", "rotation style",
                 "volume", "note", "frequency", "samples", "sample rate", "spectrum", "resolution", "snap", "motion", "turbo mode",
                 "flat line ends", "long pen vectors", "video capture", "mirror video", "year", "month", "date", "day of week",
                 "hour", "minute", "second", "time in milliseconds", "duration", "number of channels", "tempo", "volume", "balance",
                 "abs", "neg", "ceiling", "floor", "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "ln", "log", "lg", "e^", "10^", "2^",
                 "number", "text", "Boolean", "list", "sprite", "costume", "sound", "command", "reporter", "predicate", "item", "thing"]


class CodeGraph:

    # Constructor
    def __init__(self, json_file, code_shape_p_q_list):
        # default dictionary to store graph

        def converting_trees(json_code, parent_node, order):
            """Converting single json code to an AST tree.
            Input:
            json_code: tuple. json object. The json object of a snap program.
            parent_node: Node. Parent node, used as parent in this recursion.
            order: str. Stroring the current order of this node with other childrens.
            """
            # Record current node, children order.
            try:
                value = json_code['value']
                # if value in system_values:
                node = Node(json_code['type'] + "_" + json_code['value'], parent=parent_node, order=order)
                # else:
                #     node = Node(json_code['type'], parent=parent_node, order=order)
            except:
                node = Node(json_code['type'], parent=parent_node, order=order)

            # If current json part doesn't have a child, return.
            if not json_code.get('childrenOrder'):
                return

            # Recursion for every child under current node
            for child_order in json_code['childrenOrder']:
                converting_trees(json_code['children'][child_order], node, child_order)

        try:
            with open(json_file) as json_file:
                json_code = (json.load(json_file))
            self.head = Node(json_code['type'])
        except:
            print("json.loads(code) error: ", json_file)
            self.head = Node("no_code")
            return

        # Recursively construct AST tree.
        for child_order in json_code['childrenOrder']:
            print("child_order:", child_order)
            converting_trees(json_code['children'][child_order], self.head, child_order)
        # data = self.collect_all_pqgrams(code_shape_p_q_list)
        r = RenderTree(self.head)
        print(r)

    def dfs_visit_node(self, node, p, q):
        gram_list = ["empty"] * (p + q)
        if not node:
            return
        node_to_parent = copy.deepcopy(node)
        for parent_level in range(1, p):
            if node_to_parent.parent:
                gram_list[p - parent_level - 1] = node_to_parent.parent.name
                node_to_parent = node_to_parent.parent.name
        gram_list[p - 1] = node.name

        if not node.children:
            self.pqgram_set.add_pqgram(PQGram(p, q, gram_list))
            return

        if q != 0:
            children_list = [child.name for child in node.children]
            children_list = ["empty", "empty"] + children_list + ["empty", "empty"]
            for i in range(len(children_list) - 2):
                gram_list[-q:] = children_list[i:i + q]
                pqgram = PQGram(p, q, gram_list)
                self.pqgram_set.add_pqgram(pqgram)
                # print("a pqgram is being added: ", pqgram.grams, "gramlist: ", gram_list)
        elif q == 0:
            pqgram = PQGram(p, q, gram_list)
            self.pqgram_set.add_pqgram(pqgram)
            # print("a pqgram is being added: ", pqgram.grams, "gramlist: ", gram_list)

        for child in node.children:
            self.dfs_visit_node(child, p, q)

    def collect_all_pqgrams(self, code_shape_p_q_list):
        data = {}
        for p_q_list in code_shape_p_q_list:
            self.pqgram_set = PQGramSet()
            p, q = p_q_list[0], p_q_list[1]
            self.dfs_visit_node(self.head, p, q)
            for pqgram in (self.pqgram_set.pqgram_set):
                pqgram_string = str(pqgram.count) + "|" + pqgram.grams
                self.pqgram_set.pqgram_string_set.add(pqgram_string)
            code_shape = {}
            for pqgram in (self.pqgram_set.pqgram_set):
                code_shape[pqgram.grams] = (pqgram.count)
            data["code_state" + str(p_q_list)] = code_shape
        return (data)


class PQGram():
    def __init__(self, p, q, gram_list):
        self.p = p
        self.q = q
        self.grams = "|".join(gram_list)
        self.grams = str(p) + str(q) + "|" + self.grams
        self.count = 1

class PQGramSet():
    def __init__(self):
        self.pqgram_set = set()
        self.pqgram_string_set = set()
    def add_pqgram(self, next_pqgram):
        for existing_pqgram in self.pqgram_set:
            if existing_pqgram.grams == next_pqgram.grams:
                existing_pqgram.count += 1
                return
        self.pqgram_set.add(next_pqgram)


def get_json():
    file_path = root_dir + "Datasets/data/SnapJSON_413/"
    file_list = os.listdir(file_path)
    for file_name in file_list:
        if file_name.endswith(".json"):
            try:
                CodeGraph(file_path + '/' + file_name, [[2,3], [2,1]])
            except:
                print("error: ", file_name)



# json_file = "170315872.xml.json"
# codegraph = CodeGraph("/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets/keymove.json", [[2,3], [2,1]])
# data = codegraph.collect_all_pqgrams( [[2,3], [2,1]])
# print(data)


# get_json()