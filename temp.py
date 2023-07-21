import timm

m = timm.list_models(pretrained=True)
for i in m:
    print(i)