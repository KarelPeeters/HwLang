import hwl

source = hwl.SourceDatabase("../../design/project")
print("Parsed files:")
for f in source.files:
    print(f"  {f}")

parsed = source.parse()
print(parsed)

foo = parsed.resolve("top.top")
print(foo)
