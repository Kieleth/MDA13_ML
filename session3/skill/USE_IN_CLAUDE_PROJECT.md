# Cómo usar el skill en un Claude Project

1. Ve a [claude.ai/projects](https://claude.ai/projects).
2. Click **"+ New project"**.
3. Pon un nombre: "Cañadata Lead Advisor".
4. En **"Project knowledge"** o **"Custom instructions"** (según versión de la UI), pega ENTERO el contenido de `canadata_lead_advisor.md`.
5. Guarda el proyecto.
6. Abre una conversación nueva dentro del proyecto.
7. Pega un lead de prueba. El bot responde con el formato obligado.
8. Comparte el proyecto con tu colega: botón **"Share"** → URL.

## Cosas a tener en cuenta

- El bot del proyecto **no tiene acceso a tu CSV de leads**. Si quieres que use tu histórico, adjunta el CSV en "Project knowledge". Atención al tamaño: Claude permite ~200 MB pero la lectura razonable son <10 MB.
- Si tienes plan Pro, el proyecto persiste y se comparte. Sin Pro, sólo dentro de tu cuenta.
- El skill funciona en Claude Sonnet 3.7+, Opus 4, Haiku 4.5. Ajusta según presupuesto.

## Para iterarlo

Edita `canadata_lead_advisor.md` localmente. Cuando cambies algo importante, actualiza las instrucciones del proyecto pegando la nueva versión. Versiona el `.md` en git como cualquier otro código.
