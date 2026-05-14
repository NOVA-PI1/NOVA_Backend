# pyrefly: ignore [missing-import]
from crewai import Task

def create_nova_tasks(editorial_agent, ethical_agent, multimodal_agent, dialectical_agent) -> list[Task]:
    """
    Define las 4 tareas secuenciales para el flujo de trabajo de NOVA v2.0.
    """
    
    tarea_redaccion = Task(
        description=(
            "Investigar y redactar un borrador de noticia sobre el tema: {tema}. "
            "Es obligatorio utilizar la herramienta BCLTool para encontrar datos "
            "reales y contexto local antes de escribir."
        ),
        expected_output=(
            "Un borrador de noticia con título, entradilla y cuerpo, fundamentado "
            "en datos de la BCL local."
        ),
        agent=editorial_agent
    )

    tarea_revision_etica = Task(
        description=(
            "Analizar el borrador de noticia generado para detectar sesgos de género, "
            "raza o colonialismo. Corregir el lenguaje para que sea equitativo y "
            "transparente."
        ),
        expected_output=(
            "Una versión de la noticia purgada de sesgos junto con un breve reporte "
            "de los ajustes éticos realizados."
        ),
        agent=ethical_agent
    )

    tarea_visual = Task(
        description=(
            "Basado en el texto final éticamente aprobado, proponer 2 descripciones "
            "de imágenes (prompts) que apoyen la narrativa sin caer en estereotipos "
            "sobre Latinoamérica o el Sur Global."
        ),
        expected_output=(
            "Dos prompts detallados para generación de imágenes que aseguren "
            "diversidad y dignidad visual."
        ),
        agent=multimodal_agent
    )

    tarea_reflexion = Task(
        description=(
            "Analizar la noticia final y los prompts visuales para proponer dos o "
            "tres preguntas críticas o ángulos ciegos que fomenten la reflexión del "
            "periodista humano."
        ),
        expected_output=(
            "Un conjunto de 2-3 preguntas críticas finales para el periodista."
        ),
        agent=dialectical_agent
    )

    return [tarea_redaccion, tarea_revision_etica, tarea_visual, tarea_reflexion]
