"""Layout hints for invoice-first + field-ticket-second packet structures."""

PACKET_LAYOUT_RULES = {
    "invoice_page": {"likely_page_index": 0},
    "field_ticket_page": {"likely_page_index": 1},
}
