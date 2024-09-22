from generate_response import OllamaClient

def main():
    client = OllamaClient()
    prompt = "What caused the Treaty of Versailles?"
    context = "The Treaty of Versailles was primarily caused by the aftermath of World War I, where the Allied Powers sought to impose strict sanctions and reparations on Germany to prevent future aggression."
    response = client.generate_response(prompt, context)
    print("Generated Response:", response)

if __name__ == "__main__":
    main()