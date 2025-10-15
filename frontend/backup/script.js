document.addEventListener('DOMContentLoaded', () => {
    const translateButton = document.getElementById('translate-button');
    const textToTranslate = document.getElementById('text-to-translate');
    const sourceLanguage = document.getElementById('source-language');
    const outputDiv = document.getElementById('output');

    translateButton.addEventListener('click', async () => {
        const text = textToTranslate.value;
        const lang = sourceLanguage.value;

        outputDiv.innerText = "Translating...";

        if (!text.trim()) {
            outputDiv.innerText = "Please enter some text to translate.";
            return;
        }

        try {
            const response = await fetch('/translate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'accept': 'application/json'
                },
                body: JSON.stringify({
                    text: text,
                    source_language: lang
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'An error occurred');
            }

            const data = await response.json();
            outputDiv.innerText = data.translated_text;
        } catch (error) {
            outputDiv.innerText = `Error: ${error.message}`;
        }
    });
});
