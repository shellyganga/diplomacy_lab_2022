text = "Orphaned fruit bat pups nursed back to health in animal sanctuary http://t.co/BoYvGQD8WA http://t.co/tnijKVgpG3"
from ttp import ttp
p = ttp.Parser()
def get_text(text):

    text_fin = ""

    result = p.parse(text)
    print(result)

    arr = text.split(" ")
    print(arr)
    print(text_fin)
    for elem in arr:
        if ((elem not in str(result.users)) and (elem not in str(result.tags)) and (elem not in str(result.urls))):
            text_fin = text_fin  + elem + " "

    return text_fin[:-1]

print(get_text((text)))