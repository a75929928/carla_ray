# heroes = {'1': 2, '3': 4}
# for hero_id, hero in heroes.keys(), heroes.values():
#     print(hero_id, hero)

a = {'1': 2}
b = a
b.update({'1': 3})
print(a,b)