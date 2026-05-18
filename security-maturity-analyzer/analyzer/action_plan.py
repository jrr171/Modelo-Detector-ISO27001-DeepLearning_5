"""
Plan de Acción Automático
Genera un plan de acción priorizado por dominio más débil con acciones específicas,
nivel de esfuerzo y tiempo estimado de implementación.
"""

from typing import Dict, List
from analyzer.event_classifier import DomainStats
from analyzer.maturity_scorer  import MaturityResult
from rules.iso27001_controls   import ISO27001_DOMAINS, MATURITY_LEVELS


# Acciones específicas por dominio y nivel actual
DOMAIN_ACTIONS = {
    "access_control": {
        "clause": "A.9 — Control de Accesos",
        "levels": {
            (0, 20): {
                "effort": "Alto",
                "tiempo": "1–3 meses",
                "actions": [
                    "Implementar autenticación multifactor (MFA) para todos los accesos SSH y sistemas críticos.",
                    "Establecer política de contraseñas: mínimo 12 caracteres, rotación cada 90 días.",
                    "Instalar fail2ban o equivalente para bloquear IPs tras 5 intentos fallidos.",
                    "Auditar y eliminar cuentas de usuario innecesarias o con contraseñas por defecto.",
                    "Implementar logging centralizado de todos los accesos (éxitos y fallos).",
                ]
            },
            (21, 40): {
                "effort": "Medio",
                "tiempo": "2–4 semanas",
                "actions": [
                    "Centralizar los logs de autenticación en un SIEM o servidor de logs dedicado.",
                    "Configurar alertas automáticas para patrones de fuerza bruta (>5 fallos en 1 min).",
                    "Revisar y documentar la matriz de accesos por rol (RBAC).",
                    "Implementar revisión trimestral de cuentas activas y privilegios.",
                ]
            },
            (41, 60): {
                "effort": "Medio",
                "tiempo": "2–6 semanas",
                "actions": [
                    "Automatizar la revisión de accesos privilegiados con reportes mensuales.",
                    "Implementar gestión de sesiones con timeout automático (15 min inactividad).",
                    "Configurar alertas de acceso en horarios fuera de oficina.",
                ]
            },
            (61, 100): {
                "effort": "Bajo",
                "tiempo": "Mantenimiento continuo",
                "actions": [
                    "Mantener el programa de revisión periódica de accesos privilegiados.",
                    "Evaluar implementación de Zero Trust Architecture para accesos externos.",
                    "Integrar threat intelligence para detección proactiva de credenciales comprometidas.",
                ]
            },
        }
    },
    "operations_security": {
        "clause": "A.12 — Seguridad en Operaciones",
        "levels": {
            (0, 20): {
                "effort": "Alto",
                "tiempo": "1–2 meses",
                "actions": [
                    "Implementar política de backups automáticos: diario (incremental) y semanal (completo).",
                    "Establecer proceso de gestión de parches: evaluación semanal, aplicación mensual.",
                    "Instalar y configurar antivirus/EDR en todos los servidores.",
                    "Documentar un inventario de activos de software autorizado.",
                ]
            },
            (21, 40): {
                "effort": "Medio",
                "tiempo": "3–6 semanas",
                "actions": [
                    "Automatizar las actualizaciones de seguridad del sistema operativo.",
                    "Implementar monitoreo de integridad de archivos (FIM) en directorios críticos.",
                    "Verificar y documentar los procedimientos de restauración de backups.",
                ]
            },
            (41, 60): {
                "effort": "Bajo",
                "tiempo": "2–4 semanas",
                "actions": [
                    "Implementar escaneo automático de vulnerabilidades (OpenVAS, Nessus).",
                    "Establecer métricas de operaciones: tiempo de recuperación, tasa de éxito de backups.",
                ]
            },
            (61, 100): {
                "effort": "Bajo",
                "tiempo": "Mejora continua",
                "actions": [
                    "Automatizar el ciclo completo de gestión de parches con ventanas de mantenimiento.",
                    "Implementar deception technology (honeypots) para detección temprana de intrusos.",
                ]
            },
        }
    },
    "communications_security": {
        "clause": "A.13 — Seguridad en Comunicaciones",
        "levels": {
            (0, 20): {
                "effort": "Alto",
                "tiempo": "2–4 semanas",
                "actions": [
                    "Activar HTTPS con certificado SSL/TLS válido en todos los servicios web.",
                    "Forzar redirección automática de HTTP a HTTPS (HSTS).",
                    "Configurar firewall perimetral con política de denegación por defecto.",
                    "Deshabilitar protocolos inseguros: SSLv2, SSLv3, TLS 1.0, TLS 1.1.",
                ]
            },
            (21, 40): {
                "effort": "Medio",
                "tiempo": "2–4 semanas",
                "actions": [
                    "Implementar VPN para accesos remotos de administradores.",
                    "Segmentar la red por zonas (DMZ, intranet, producción, administración).",
                    "Configurar IDS/IPS para detectar escaneos y ataques de red.",
                ]
            },
            (41, 60): {
                "effort": "Medio",
                "tiempo": "1–3 semanas",
                "actions": [
                    "Implementar monitoreo de ancho de banda para detectar exfiltración de datos.",
                    "Revisar y actualizar las reglas del firewall semestralmente.",
                ]
            },
            (61, 100): {
                "effort": "Bajo",
                "tiempo": "Mejora continua",
                "actions": [
                    "Implementar microsegmentación de red para sistemas críticos de Comercio Exterior.",
                    "Evaluar SD-WAN para mejorar la seguridad en conexiones con SUNAT/Aduanet.",
                ]
            },
        }
    },
    "incident_management": {
        "clause": "A.16 — Gestión de Incidentes",
        "levels": {
            (0, 20): {
                "effort": "Alto",
                "tiempo": "1–2 meses",
                "actions": [
                    "Redactar y publicar el Plan de Respuesta a Incidentes de Seguridad (PRIS).",
                    "Designar un equipo de respuesta a incidentes con roles y responsabilidades claras.",
                    "Implementar un sistema básico de tickets para registrar incidentes (Jira, Redmine).",
                    "Establecer canales de comunicación de emergencia para incidentes críticos.",
                    "Definir los criterios de clasificación de incidentes (bajo/medio/alto/crítico).",
                ]
            },
            (21, 40): {
                "effort": "Medio",
                "tiempo": "3–5 semanas",
                "actions": [
                    "Crear playbooks de respuesta para los 5 tipos de incidente más frecuentes.",
                    "Realizar simulacro de respuesta a incidente al menos una vez al año.",
                    "Establecer métricas: MTTD (tiempo medio de detección) y MTTR (tiempo de resolución).",
                ]
            },
            (41, 60): {
                "effort": "Medio",
                "tiempo": "2–4 semanas",
                "actions": [
                    "Automatizar la creación de tickets de incidente desde alertas del SIEM.",
                    "Implementar análisis post-mortem obligatorio para incidentes de severidad alta.",
                ]
            },
            (61, 100): {
                "effort": "Bajo",
                "tiempo": "Mejora continua",
                "actions": [
                    "Integrar threat intelligence feeds para enriquecer la respuesta a incidentes.",
                    "Implementar SOAR (Security Orchestration, Automation and Response).",
                ]
            },
        }
    },
    "cryptography": {
        "clause": "A.10 — Criptografía",
        "levels": {
            (0, 20): {
                "effort": "Alto",
                "tiempo": "2–4 semanas",
                "actions": [
                    "Cifrar todas las bases de datos que contengan información sensible (AES-256).",
                    "Renovar certificados SSL/TLS expirados e implementar renovación automática (Let's Encrypt).",
                    "Eliminar el uso de MD5 y SHA-1; migrar a SHA-256 o superior.",
                    "Implementar política de gestión de claves criptográficas documentada.",
                ]
            },
            (21, 40): {
                "effort": "Medio",
                "tiempo": "2–3 semanas",
                "actions": [
                    "Implementar cifrado de extremo a extremo para comunicaciones con SUNAT/Aduanet.",
                    "Establecer rotación periódica de claves de API y certificados (máximo 1 año).",
                ]
            },
            (41, 60): {
                "effort": "Bajo",
                "tiempo": "1–2 semanas",
                "actions": [
                    "Auditar el inventario de certificados y claves activas con herramientas automatizadas.",
                    "Implementar HSM (Hardware Security Module) para claves de alto valor.",
                ]
            },
            (61, 100): {
                "effort": "Bajo",
                "tiempo": "Mejora continua",
                "actions": [
                    "Evaluar implementación de criptografía post-cuántica para sistemas críticos.",
                    "Mantener el proceso de auditoría criptográfica semestral.",
                ]
            },
        }
    },
    "physical_security": {
        "clause": "A.11 — Seguridad Física",
        "levels": {
            (0, 20): {
                "effort": "Alto",
                "tiempo": "1–3 meses",
                "actions": [
                    "Instalar control de acceso físico (tarjeta o PIN) al datacenter/sala de servidores.",
                    "Implementar UPS (Sistema de Alimentación Ininterrumpida) con autonomía mínima de 4 horas.",
                    "Instalar cámaras de seguridad en accesos al datacenter y servidores.",
                    "Establecer política de dispositivos removibles (bloquear USB no autorizados).",
                ]
            },
            (21, 40): {
                "effort": "Medio",
                "tiempo": "2–4 semanas",
                "actions": [
                    "Implementar monitoreo ambiental: temperatura, humedad y detección de incendios.",
                    "Establecer registro de acceso físico al datacenter con log de entradas y salidas.",
                ]
            },
            (41, 60): {
                "effort": "Bajo",
                "tiempo": "2–3 semanas",
                "actions": [
                    "Implementar autenticación multifactor para acceso físico (tarjeta + PIN o biometría).",
                    "Revisar y actualizar la política de escritorio limpio y pantalla bloqueada.",
                ]
            },
            (61, 100): {
                "effort": "Bajo",
                "tiempo": "Mejora continua",
                "actions": [
                    "Realizar auditoría física anual de controles de acceso y equipos.",
                    "Implementar sistema de gestión de visitantes con registro fotográfico.",
                ]
            },
        }
    },
}


def get_level_range(score: float) -> tuple:
    if score <= 20:  return (0, 20)
    if score <= 40:  return (21, 40)
    if score <= 60:  return (41, 60)
    return (61, 100)


def generate_action_plan(result: MaturityResult) -> List[dict]:
    """Generate prioritized action plan sorted by weakest domain first."""
    plan = []
    sorted_domains = sorted(result.domain_scores.items(), key=lambda x: x[1].raw_score)

    for priority, (key, ds) in enumerate(sorted_domains, 1):
        if ds.raw_score >= 85:
            continue  # Skip domains already optimized
        domain_info  = DOMAIN_ACTIONS.get(key, {})
        level_range  = get_level_range(ds.raw_score)
        level_data   = domain_info.get("levels", {}).get(level_range, {})
        actions      = level_data.get("actions", ["Mantener y mejorar los controles existentes."])
        effort       = level_data.get("effort", "Medio")
        tiempo       = level_data.get("tiempo", "Variable")

        next_threshold = min(100, (int(ds.raw_score / 20) + 1) * 20)
        gap = next_threshold - ds.raw_score

        plan.append({
            "priority": priority,
            "domain":   domain_info.get("clause", ds.domain_name),
            "score":    ds.raw_score,
            "level":    ds.level,
            "level_name": ds.level_name,
            "gap_to_next": round(gap, 1),
            "next_threshold": next_threshold,
            "actions":  actions,
            "effort":   effort,
            "tiempo":   tiempo,
        })

    return plan
