# Cómo usar el skill como Custom GPT

1. Ve a [chat.openai.com/gpts](https://chat.openai.com/gpts) (necesitas plan ChatGPT Plus).
2. Click **"+ Create"**.
3. En el panel **"Configure"** (no en "Create" del chat conversacional):
   - **Name**: "Cañadata Lead Advisor".
   - **Description**: "Asesor de leads B2B B2B de Cañadata".
   - **Instructions**: pega ENTERO el contenido de `canadata_lead_advisor.md`.
   - **Capabilities**: desactiva "Web Browsing", "DALL·E", "Code Interpreter" (no los necesitas).
4. **Save** → "Only me" si todavía es prueba, o "Anyone with the link" / "Public" para compartir.
5. Abre el GPT y pega un lead de prueba.

## Diferencias vs Claude Project

- Custom GPT permite **Actions** (function calling via HTTP a tu API). Si quieres conectar tu CRM o tu clasificador entrenado, puedes hacer que el GPT haga POST a un endpoint tuyo. Esto te acerca al paso_7 local pero ya con UI hosted por OpenAI.
- Si vas a usar Actions, tendrás que exponer un endpoint (FastAPI, lambda, etc.) y darle al GPT su OpenAPI schema. Es la siguiente capa de complejidad, fuera del scope de este skill básico.

## Para iterarlo

Edita el `.md` local. Cuando quieras actualizar el GPT, copia y pega de nuevo las instructions. ChatGPT no tiene versioning automático del GPT; lleva tú el control en git.
