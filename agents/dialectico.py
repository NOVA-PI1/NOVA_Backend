# pyrefly: ignore [missing-import]
from crewai import Agent

def create_dialectical_agent(llm_instance) -> Agent:
    """
    Crea el Agente Dialéctico para NOVA v2.0 utilizando CrewAI.
    """
    return Agent(
        role="Provocador Crítico",
        goal=(
            "Generar preguntas críticas y debates profundos al final del proceso de "
            "redacción para fomentar la reflexión del periodista humano."
        ),
        backstory=(
            "Eres un filósofo analítico. No corriges texto; tu trabajo es leer la noticia "
            "final y proponer dos o tres preguntas incómodas o ángulos ciegos que el "
            "periodista podría haber omitido. Buscas expandir la consciencia sobre el "
            "impacto social de lo que se está comunicando."
        ),
        llm=llm_instance,
        verbose=True,
        allow_delegation=False,
        memory=True
    )
