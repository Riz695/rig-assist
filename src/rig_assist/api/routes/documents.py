from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from fastapi import Path as PathParam
from qdrant_client.http.exceptions import ResponseHandlingException

from rig_assist.api.dependencies import get_app_settings, get_search_index, qdrant_unavailable
from rig_assist.api.schemas import DocumentList, DocumentResponse
from rig_assist.config import Settings
from rig_assist.ingestion.cache import PageCache
from rig_assist.ingestion.files import write_bytes_atomic
from rig_assist.ingestion.page_images import render_page_png
from rig_assist.ingestion.parsers.olmocr import OcrFailedError
from rig_assist.ingestion.pipeline import ingest_file, make_doc_id, make_ocr_client
from rig_assist.ingestion.store import DocumentInfo
from rig_assist.retrieval.service import SearchIndex

router = APIRouter(tags=["documents"])


def _to_response(info: DocumentInfo) -> DocumentResponse:
    return DocumentResponse(
        doc_id=info.doc_id,
        title=info.title,
        filename=info.filename,
        pages=info.pages,
        chunks=info.chunks,
        missing_pages=info.missing_pages,
    )


@router.get("/documents", summary="List indexed documents")
def list_documents(index: Annotated[SearchIndex, Depends(get_search_index)]) -> DocumentList:
    return DocumentList(documents=[_to_response(info) for info in index.store.documents()])


@router.post(
    "/documents",
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a PDF manual",
    responses={
        409: {"description": "A different file already uses this doc_id"},
        415: {"description": "The file is not a PDF"},
        502: {"description": "OCR is configured but the OCR server failed"},
        503: {"description": "Qdrant is unreachable, so the document can't be indexed"},
    },
)
def upload_document(
    file: Annotated[UploadFile, File(description="A PDF manual.")],
    settings: Annotated[Settings, Depends(get_app_settings)],
    index: Annotated[SearchIndex, Depends(get_search_index)],
) -> DocumentResponse:
    """Save the PDF to the raw folder, parse (OCR per OCR_MODE), chunk and index it
    (BM25 store and Qdrant).

    Uploading the same file name again replaces that document. Runs synchronously:
    a 250-page manual takes ~15 s to parse without OCR, plus ~0.2 s per chunk to embed.
    """
    # Keep only the base name: "../../x.pdf" must not escape the raw folder.
    name = Path(file.filename or "").name
    data = file.file.read()
    if not name.lower().endswith(".pdf") or not data.startswith(b"%PDF-"):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only PDF files are accepted.")
    try:
        doc_id = make_doc_id(name)
    except ValueError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error

    raw_dir = settings.data_raw_dir
    clashes = (
        [
            p.name
            for p in raw_dir.glob("*")
            if p.suffix.lower() == ".pdf" and p.name != name and make_doc_id(p.name) == doc_id
        ]
        if raw_dir.exists()
        else []
    )
    if clashes:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"doc_id '{doc_id}' is already used by {clashes[0]}; rename one of the files.",
        )

    path = raw_dir / name
    write_bytes_atomic(path, data)
    try:
        info = ingest_file(
            path,
            index.store,
            ocr_mode=settings.ocr_mode,
            ocr_client=make_ocr_client(settings),
            cache=PageCache(settings.data_parsed_dir),
            ocr_concurrency=settings.ocr_concurrency,
            dense=index.dense,
        )
    except ValueError as error:  # e.g. a damaged PDF that starts with %PDF- but can't be read
        path.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(error)) from error
    except OcrFailedError as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error)) from error
    except ResponseHandlingException as error:
        # The PDF stays in the raw folder, unlisted: retrying the upload (or `make ingest`)
        # once Qdrant is up indexes it.
        raise qdrant_unavailable(error) from error
    index.refresh()
    return _to_response(info)


@router.get(
    "/documents/{doc_id}/pages/{page}",
    summary="One page of a manual as a PNG image (for citations)",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}, "description": "The page, 1000 px wide."},
        404: {"description": "Unknown document, page past the end, or the PDF is gone"},
    },
)
def page_image(
    doc_id: str,
    page: Annotated[int, PathParam(ge=1, description="1-based page in the PDF.")],
    settings: Annotated[Settings, Depends(get_app_settings)],
    index: Annotated[SearchIndex, Depends(get_search_index)],
) -> Response:
    """Render a cited page, so a reader can check the answer against the real manual."""
    info = index.store.document(doc_id)
    if info is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No document with ID {doc_id!r}.")
    if page > info.pages:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"{doc_id!r} has {info.pages} pages; no page {page}."
        )
    pdf_path = settings.data_raw_dir / info.filename
    if not pdf_path.is_file():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"The PDF for {doc_id!r} is no longer on the server."
        )
    png = render_page_png(pdf_path, info.file_sha256, page)
    # A page only changes when the manual is re-uploaded: let the browser keep it an hour.
    return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=3600"})
