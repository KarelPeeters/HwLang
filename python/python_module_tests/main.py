import hwl

s = hwl.Sources("../../design/project")
print("Parsed files:")
for f in s.files:
    print(f"  {f}")

