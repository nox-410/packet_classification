# -*- coding: utf-8 -*-

from linar_tc import Init_rule_set
import numpy as np
from math import ceil

from tree.hicuts import HiCuts
from tree.efficuts import EffiCuts
from tree.tree import *


class Rule_Mapper():
    def __init__(self,mode,path,num_group):
        assert(mode in ("hicut","efficut"))
        
        rules = Init_rule_set(path)
        rules[:,5:] += 1
        rules = rules[:,[0,5,1,6,2,7,3,8,4,9]]
        self.num_rule = rules.shape[0]
        self.rules = list(map(Rule,range(self.num_rule),rules))
        self.mark = np.zeros(self.num_rule)
        self.nodecnt = 0
        self.current_mark = 1
        self.rulecnt = 0
        self.mode = mode
        self.num_group = num_group
        self.num_rules_per_group = ceil(self.num_rule / num_group)
        if self.mode == "hicut":
            self.run_hicuts()
        elif self.mode == "efficut":
            self.run_efficuts()
        self.mark = self.mark.astype(np.int32)
        self.check_map()
        
    def _dfs(self, tree, x):
        self.nodecnt += 1
        if tree.is_leaf(x) is True:
            for rule in x.rules:
                if self.mark[rule.priority] == 0:
                    self.mark[rule.priority] = self.current_mark
                    self.rulecnt += 1
                    if self.rulecnt == self.num_rules_per_group:# Remember to alter this line if you need other than 500*200 classification.
                        self.rulecnt, self.current_mark = 0, self.current_mark + 1
        for y in x.children:
            self._dfs(tree, y)
        return self.rulecnt
    
    def run_hicuts(self):
        cuts = HiCuts(self.rules)
        cuts.train()
        self._dfs(cuts.tree, cuts.tree.root)
        
    def run_efficuts(self):
        cutt = EffiCuts(self.rules)
        cutt.train()
        for tree in cutt.trees:
            print("depth=", tree.depth, "rules=", len(tree.root.rules))
            self.nodecnt = 0
            self._dfs(tree, tree.root)
            print("cnt=", self.nodecnt)

    def check_map(self):
        assert(self.mark.min() >= 0)
        assert(self.mark.max() <= self.num_group)
        

if __name__ == "__main__":
    rule_number = 1000
    s = Rule_Mapper("hicut",
                    "data/rule_{0}.rule".format(rule_number),
                    num_group = 50)
        