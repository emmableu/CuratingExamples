library("arulesViz")
data(Groceries)
rules <- apriori(Groceries, parameter=list(support=0.005, confidence=0.5))
# View(rules)
summary(Groceries)