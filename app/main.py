
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "http://localhost:11434/v1",
)
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")

LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2200"))

MAX_GENERATION_ATTEMPTS = 2
BATCH_MAX_SIZE = 20

SEO_TITLE_MAX_LENGTH = 70
SEO_DESCRIPTION_MIN_LENGTH = 800
SEO_DESCRIPTION_MAX_LENGTH = 1200


# ============================================================
# 2. LOGGING
# ============================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("product_generator")


# ============================================================
# 3. PYDANTIC MODELS
# ============================================================

class ProductInput(BaseModel):
    """Исходные данные о товаре."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=300)
    manufacturer: str | None = Field(default=None, max_length=300)
    specs: dict[str, Any] = Field(default_factory=dict)
    description: str | None = Field(default=None, max_length=5000)


class DraftOutput(BaseModel):
    """Сгенерированная часть карточки товара."""

    seo_title: str
    seo_description: str
    search_queries: list[str]
    commercial_block: str
    missing_fields: list[str]


class ProductResponse(BaseModel):
    product: ProductInput
    characteristics: dict[str, Any]
    draft: DraftOutput
    quality_checks: dict[str, Any]


class BatchRequest(BaseModel):
    products: list[ProductInput] = Field(
        min_length=1,
        max_length=BATCH_MAX_SIZE,
    )


class BatchResponse(BaseModel):
    results: list[ProductResponse]


# ============================================================
# 4. APPLICATION LIFECYCLE
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application started")
    logger.info("LLM model: %s", LLM_MODEL)
    logger.info("LLM base URL: %s", LLM_BASE_URL)

    yield

    logger.info("Application stopped")


app = FastAPI(
    title="AI Product Description Generator",
    description=(
        "Генерация SEO-карточек технических товаров "
        "с использованием локальной или удалённой LLM."
    ),
    version="3.0.0",
    lifespan=lifespan,
)


# ============================================================
# 5. REQUEST LOGGING
# ============================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    started_at = time.perf_counter()

    try:
        response = await call_next(request)

        elapsed = time.perf_counter() - started_at
        response.headers["X-Request-ID"] = request_id

        logger.info(
            "[%s] %s %s -> %s (%.2fs)",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )

        return response

    except Exception:
        logger.exception("[%s] Request failed", request_id)
        raise


# ============================================================
# 6. IMPORTANT PRODUCT FIELDS
# ============================================================

IMPORTANT_FIELDS = {
    "Материал корпуса": [
        "материал корпуса",
    ],
    "Класс защиты": [
        "класс защиты",
        "степень защиты",
        "ip",
    ],
    "Рабочая температура": [
        "рабочая температура",
        "температура эксплуатации",
        "температурный диапазон",
    ],
    "Комплектация": [
        "комплектация",
        "комплект поставки",
    ],
    "Гарантийный срок": [
        "гарантийный срок",
        "срок гарантии",
        "гарантия",
    ],
    "Стандарты и сертификаты": [
        "стандарт",
        "сертификат",
        "гост",
        "ту",
    ],
    "Материалы внутренних деталей": [
        "материал внутренних деталей",
        "материал механизма",
        "материал мембраны",
        "материал чувствительного элемента",
    ],
}


def normalize_text(value: Any) -> str:
    """Нормализует текст для сопоставления названий полей."""

    text = str(value).strip().lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text)


def field_matches_alias(
    provided_key: str,
    alias: str,
) -> bool:
    """Проверяет соответствие названия характеристики алиасу."""

    if alias == "ip":
        # Не допускаем совпадений с IP внутри других слов.
        return bool(
            re.search(
                r"(^|[\s_-])ip($|[\s_-])",
                provided_key,
            )
        )

    return (
        provided_key == alias
        or provided_key.startswith(alias + " ")
        or provided_key.startswith(alias + " (")
        or provided_key.startswith(alias + "_")
        or provided_key.startswith(alias + "-")
    )


def calculate_missing_fields(
    product: ProductInput,
) -> list[str]:
    """
    Определяет отсутствующие важные характеристики.

    Проверяются названия полей specs, а не ответ LLM.
    """

    provided_keys = [
        normalize_text(key)
        for key in product.specs.keys()
    ]

    missing = []

    for field_name, aliases in IMPORTANT_FIELDS.items():
        normalized_aliases = [
            normalize_text(alias)
            for alias in aliases
        ]

        is_provided = any(
            field_matches_alias(provided_key, alias)
            for provided_key in provided_keys
            for alias in normalized_aliases
        )

        if not is_provided:
            missing.append(field_name)

    return missing


# ============================================================
# 7. COMMERCIAL BLOCK
# ============================================================

def build_commercial_block(
    product: ProductInput,
) -> str:
    """
    Формирует коммерческий блок только из известных фактов.

    LLM не может самостоятельно добавлять в него
    неподтверждённые преимущества и области применения.
    """

    parts = [product.name.strip()]

    if product.manufacturer:
        parts.append(
            f"производитель — {product.manufacturer.strip()}"
        )

    if product.description:
        description = product.description.strip()

        if description:
            parts.append(description)

    if product.specs:
        specs_parts = [
            f"{key.lower()} — {value}"
            for key, value in product.specs.items()
            if value is not None and str(value).strip()
        ]

        if specs_parts:
            parts.append(
                "Основные характеристики: "
                + "; ".join(specs_parts)
            )

    result = ". ".join(parts).strip()

    if result and result[-1] not in ".!?":
        result += "."

    return result


# ============================================================
# 8. PROMPT CREATION
# ============================================================

def make_prompt(
    product: ProductInput,
    retry_feedback: str = "",
) -> str:
    """Формирует prompt для генерации SEO-контента."""

    product_data = product.model_dump()

    return f"""
Ты — редактор технического B2B-каталога.

Подготовь SEO-контент для карточки измерительного прибора.

ГЛАВНОЕ ПРАВИЛО:
Используй только факты, явно присутствующие во входных данных.

ЗАПРЕЩЕНО:
- Придумывать технические характеристики.
- Придумывать области применения и отрасли.
- Придумывать условия эксплуатации.
- Придумывать материалы, сертификаты и стандарты.
- Придумывать совместимость и комплектацию.
- Обещать надёжность, срок службы или преимущества,
  если они не подтверждены входными данными.
- Добавлять числовые значения, отсутствующие во входных данных.
- Выдавать предположения за факты.

SEO TITLE:
- Не более 70 символов.
- Используй название товара.
- Производителя добавляй, если это уместно.

SEO DESCRIPTION:
- Целевая длина: 800–1200 символов.
- Пиши естественно и информативно.
- Используй только подтверждённые факты.
- Не повторяй одну информацию ради увеличения длины.
- Не добавляй общие рекламные обещания.
- Если входных данных недостаточно для 800 символов,
  напиши максимально содержательный текст меньшей длины.
- Фактическая точность важнее целевой длины.

SEARCH QUERIES:
- От 3 до 5 запросов.
- Запросы должны относиться к этому товару.
- Используй название, производителя и известные характеристики.
- Не добавляй неподтверждённые параметры.
- Избегай практически одинаковых запросов.

Верни только JSON без Markdown и пояснений.

Структура ответа:

{{
    "seo_title": "Название товара",
    "seo_description": "SEO-описание",
    "search_queries": [
        "запрос 1",
        "запрос 2",
        "запрос 3"
    ]
}}

ВХОДНЫЕ ДАННЫЕ:

{json.dumps(product_data, ensure_ascii=False, indent=2)}

{retry_feedback}
"""


# ============================================================
# 9. JSON PARSING
# ============================================================

def parse_json_response(
    raw_content: str,
) -> dict[str, Any]:
    """Извлекает JSON-объект из ответа модели."""

    content = raw_content.strip()

    if content.startswith("```"):
        content = re.sub(
            r"^```(?:json)?\s*",
            "",
            content,
            flags=re.IGNORECASE,
        )
        content = re.sub(r"\s*```$", "", content)

    try:
        result = json.loads(content)

    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")

        if start == -1 or end == -1:
            raise ValueError(
                "В ответе модели не найден JSON-объект"
            )

        result = json.loads(content[start:end + 1])

    if not isinstance(result, dict):
        raise ValueError(
            "Ответ модели должен быть JSON-объектом"
        )

    return result


# ============================================================
# 10. CONTENT QUALITY VALIDATION
# ============================================================

UNSUPPORTED_CLAIMS = [
    r"высокая надежность",
    r"повышенная надежность",
    r"длительный срок службы",
    r"устойчив к коррозии",
    r"коррозионная стойкость",
    r"экстремальные условия",
    r"широкий спектр применения",
    r"подходит для любых",
    r"гарантирует точность",
    r"обеспечивает точность",
    r"высокая точность",
    r"нефтяной и газовой промышленности",
    r"системах водоснабжения",
    r"различных технологических процессах",
]


def validate_content(
    product: ProductInput,
    draft: DraftOutput,
) -> dict[str, Any]:
    """
    Выполняет эвристические проверки контента.

    Эти проверки помогают обнаруживать потенциальные проблемы,
    но не гарантируют полного отсутствия галлюцинаций.
    """

    source_text = " ".join(
        [
            product.name or "",
            product.manufacturer or "",
            product.description or "",
            *[str(key) for key in product.specs.keys()],
            *[str(value) for value in product.specs.values()],
        ]
    )

    generated_text = " ".join(
        [
            draft.seo_title,
            draft.seo_description,
            *draft.search_queries,
        ]
    )

    unsupported_phrases = []

    for pattern in UNSUPPORTED_CLAIMS:
        if re.search(
            pattern,
            generated_text,
            flags=re.IGNORECASE,
        ):
            unsupported_phrases.append(pattern)

    source_numbers = set(
        re.findall(
            r"\d+(?:[.,]\d+)?",
            source_text,
        )
    )

    generated_numbers = set(
        re.findall(
            r"\d+(?:[.,]\d+)?",
            generated_text,
        )
    )

    unknown_numbers = sorted(
        generated_numbers - source_numbers
    )

    title_length = len(draft.seo_title)
    description_length = len(draft.seo_description)
    queries_count = len(draft.search_queries)

    return {
        "seo_title_length": title_length,
        "seo_title_valid": (
            0 < title_length <= SEO_TITLE_MAX_LENGTH
        ),
        "seo_description_length": description_length,
        "seo_description_length_valid": (
            SEO_DESCRIPTION_MIN_LENGTH
            <= description_length
            <= SEO_DESCRIPTION_MAX_LENGTH
        ),
        "seo_description_target_not_met": (
            description_length < SEO_DESCRIPTION_MIN_LENGTH
        ),
        "search_queries_count": queries_count,
        "search_queries_valid": 3 <= queries_count <= 5,
        "unsupported_phrases": unsupported_phrases,
        "unknown_numbers": unknown_numbers,
        "no_detected_unsupported_phrases": (
            not unsupported_phrases
        ),
        "no_unknown_numbers": not unknown_numbers,
    }


def get_quality_issues(
    checks: dict[str, Any],
) -> list[str]:
    """
    Определяет проблемы, которые имеет смысл исправлять
    повторной генерацией.

    Недостаточная длина SEO description сама по себе
    не вызывает повторную генерацию.
    """

    issues = []

    if not checks["seo_title_valid"]:
        issues.append(
            "SEO title должен содержать от 1 до 70 символов"
        )

    if not checks["search_queries_valid"]:
        issues.append(
            "Нужно подготовить от 3 до 5 поисковых запросов"
        )

    if checks["unsupported_phrases"]:
        issues.append(
            "Обнаружены потенциально неподтверждённые утверждения"
        )

    if checks["unknown_numbers"]:
        issues.append(
            "Обнаружены числовые значения, отсутствующие во входных данных"
        )

    return issues


# ============================================================
# 11. LLM REQUEST
# ============================================================

async def request_llm(
    client: AsyncOpenAI,
    prompt: str,
    request_id: str,
    attempt: int,
) -> dict[str, Any]:
    """Отправляет запрос к OpenAI-compatible API."""

    started_at = time.perf_counter()

    logger.info(
        "[%s] LLM request started | attempt=%s | model=%s",
        request_id,
        attempt,
        LLM_MODEL,
    )

    response = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты редактор технического B2B-каталога. "
                    "Отвечай только корректным JSON. "
                    "Не выдумывай факты."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.1,
        max_tokens=MAX_TOKENS,
    )

    elapsed = time.perf_counter() - started_at

    logger.info(
        "[%s] LLM request completed in %.2fs",
        request_id,
        elapsed,
    )

    if not response.choices:
        raise ValueError(
            "Модель вернула пустой список choices"
        )

    content = response.choices[0].message.content

    if not content:
        raise ValueError(
            "Модель вернула пустой ответ"
        )

    if response.usage:
        logger.info(
            "[%s] Tokens: prompt=%s completion=%s total=%s",
            request_id,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response.usage.total_tokens,
        )

    return parse_json_response(content)


# ============================================================
# 12. GENERATION PIPELINE
# ============================================================

async def generate_one(
    product: ProductInput,
    request_id: str = "no-request-id",
) -> ProductResponse:
    """Генерирует и проверяет карточку одного товара."""

    started_at = time.perf_counter()

    client = AsyncOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        timeout=LLM_TIMEOUT,
        max_retries=0,
    )

    try:
        retry_feedback = ""
        draft = None
        quality_checks = None

        for attempt in range(
            1,
            MAX_GENERATION_ATTEMPTS + 1,
        ):
            prompt = make_prompt(
                product=product,
                retry_feedback=retry_feedback,
            )

            model_data = await request_llm(
                client=client,
                prompt=prompt,
                request_id=request_id,
                attempt=attempt,
            )

            seo_title = model_data.get("seo_title")
            seo_description = model_data.get("seo_description")
            search_queries = model_data.get("search_queries")

            if not isinstance(seo_title, str):
                raise ValueError(
                    "В ответе модели отсутствует корректный seo_title"
                )

            if not isinstance(seo_description, str):
                raise ValueError(
                    "В ответе модели отсутствует корректный seo_description"
                )

            if not isinstance(search_queries, list):
                raise ValueError(
                    "В ответе модели search_queries должен быть списком"
                )

            search_queries = [
                query.strip()
                for query in search_queries
                if isinstance(query, str) and query.strip()
            ]

            draft = DraftOutput(
                seo_title=seo_title.strip(),
                seo_description=seo_description.strip(),
                search_queries=search_queries,
                commercial_block=build_commercial_block(product),
                missing_fields=calculate_missing_fields(product),
            )

            quality_checks = validate_content(
                product=product,
                draft=draft,
            )

            issues = get_quality_issues(quality_checks)

            logger.info(
                "[%s] Quality checks: %s",
                request_id,
                json.dumps(
                    quality_checks,
                    ensure_ascii=False,
                ),
            )

            if not issues:
                break

            if attempt < MAX_GENERATION_ATTEMPTS:
                retry_feedback = (
                    "\n\nИСПРАВЬ СЛЕДУЮЩИЕ ПРОБЛЕМЫ:\n- "
                    + "\n- ".join(issues)
                    + "\n\nНе добавляй неподтверждённые факты. "
                    "Верни только JSON."
                )

                logger.warning(
                    "[%s] Retrying generation: %s",
                    request_id,
                    "; ".join(issues),
                )

            else:
                logger.warning(
                    "[%s] Quality issues remain after retry: %s",
                    request_id,
                    "; ".join(issues),
                )

        if draft is None or quality_checks is None:
            raise ValueError(
                "Не удалось сформировать результат генерации"
            )

        total_time = time.perf_counter() - started_at

        logger.info(
            "[%s] Generation finished in %.2fs",
            request_id,
            total_time,
        )

        return ProductResponse(
            product=product,
            characteristics=product.specs,
            draft=draft,
            quality_checks=quality_checks,
        )

    except APITimeoutError:
        logger.exception("[%s] LLM timeout", request_id)
        raise HTTPException(
            status_code=504,
            detail="Превышено время ожидания ответа LLM",
        )

    except APIConnectionError:
        logger.exception(
            "[%s] LLM connection error",
            request_id,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Не удалось подключиться к LLM. "
                "Проверьте доступность сервера модели."
            ),
        )

    except RateLimitError:
        logger.exception("[%s] LLM rate limit", request_id)
        raise HTTPException(
            status_code=429,
            detail="Превышен лимит запросов к LLM",
        )

    except APIStatusError as exc:
        logger.exception(
            "[%s] LLM API status error: %s",
            request_id,
            exc.status_code,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка API модели: HTTP {exc.status_code}",
        )

    except APIError:
        logger.exception("[%s] LLM API error", request_id)
        raise HTTPException(
            status_code=502,
            detail="Ошибка при обращении к API модели",
        )

    except ValueError as exc:
        logger.exception(
            "[%s] Invalid model response",
            request_id,
        )
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "[%s] Unexpected generation error",
            request_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Внутренняя ошибка генерации",
        )

    finally:
        await client.close()


# ============================================================
# 13. HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": LLM_MODEL,
    }


# ============================================================
# 14. GENERATE ONE PRODUCT
# ============================================================

@app.post(
    "/generate",
    response_model=ProductResponse,
)
async def generate_product(
    product: ProductInput,
    request: Request,
):
    request_id = request.state.request_id

    logger.info(
        "[%s] Generate product: %s",
        request_id,
        product.name,
    )

    return await generate_one(
        product=product,
        request_id=request_id,
    )


# ============================================================
# 15. BATCH GENERATION
# ============================================================

@app.post(
    "/generate/batch",
    response_model=BatchResponse,
)
async def generate_batch(
    batch: BatchRequest,
    request: Request,
):
    request_id = request.state.request_id

    logger.info(
        "[%s] Batch generation started | count=%s",
        request_id,
        len(batch.products),
    )

    results = []

    # Последовательная обработка снижает нагрузку
    # на локальную модель и объём используемой памяти.
    for index, product in enumerate(
        batch.products,
        start=1,
    ):
        item_request_id = f"{request_id}-{index}"

        result = await generate_one(
            product=product,
            request_id=item_request_id,
        )

        results.append(result)

    logger.info(
        "[%s] Batch generation completed",
        request_id,
    )

    return BatchResponse(results=results)
