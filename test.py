import json

# some JSON:
# x = {"a": 0}
# y = {"a": 0}
# x["a"] = y
# y["a"] = 1

# parse x:
y = json.loads(open("settings_CSRT.json", "r").read())
print(y)
