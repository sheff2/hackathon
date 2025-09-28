// test.ts
import { startSession, ask } from "./adkClient";

(async () => {
  console.log("Creating session...");
  await startSession();
  console.log("Session created successfully");
  console.log("Running query...");
  const reply = await ask("umiami to miami beach");
  console.log("Agent reply:\n", reply);
})();
