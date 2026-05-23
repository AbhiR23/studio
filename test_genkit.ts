import { ai } from './src/ai/genkit';

async function test() {
  try {
    const promptParts = [
      { text: "Describe this image." },
      { media: { url: "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAAAAAAAD/2wBDAAoHBwgHBgoICAgLCgoLDhgQDg0NDh0VFhEYIx8lJCIfIiEmKzcvJik0KSEiMEExNDk7Pj4+JS5ESUM8SDc9Pjv/2wBDAQoLCw4NDhwQEBw7KCIoOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozv/wAARCABQAFADASIAAhEBAxEB/8QAFwABAQEBAAAAAAAAAAAAAAAAAAECA//EACMQAAICAQQBBQEAAAAAAAAAAAABAhEQEiExQRNRYXGBofD/xAAWAQEBAQAAAAAAAAAAAAAAAAAAAQL/xAAWEQEBAQAAAAAAAAAAAAAAAAAAEQH/2gAMAwEAAhEDEQA/APfQAAwAAAAAAAAAAAAACgEABQAAAAAAAAAAAAAAoBAAUAAAAAAAAAAAAAAKAQAFAAAAAAAAAAAAAACgEABQAAH/2Q==" } }
    ];
    const response = await ai.generate({
      prompt: promptParts,
    });
    console.log("SUCCESS:", response.text);
  } catch (err: any) {
    console.error("ERROR:", err);
  }
}

test();
