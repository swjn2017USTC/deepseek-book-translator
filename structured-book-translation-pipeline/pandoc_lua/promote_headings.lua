-- P08 Lua filter 2/3 (promote_headings)
--
-- Mirrors the legacy heading-promotion regex in publish.py `clean_for_latex`
-- (r"^(#{2,6})(\s+)", MULTILINE, regex #1) at the AST level: render.py emits
-- the book title as H1 and chapters/sections as H2/H3, so after the title
-- page is dropped every remaining heading is promoted one level (H2 -> H1
-- -> \chapter, H3 -> H2 -> \section, ... up to H6 -> H5).
--
-- The legacy regex only matches heading lines that start at column 0 of the
-- raw markdown. Pandoc therefore represents those headings as direct
-- children of the document (or of a fenced Div). Headings nested inside
-- block quotes or lists are deliberately NOT promoted, keeping the Lua path
-- byte-identical to the legacy path for such shapes.

local function promote(blocks)
  for _, block in ipairs(blocks) do
    if block.t == "Header" and block.level >= 2 and block.level <= 6 then
      block.level = block.level - 1
    elseif block.t == "Div" then
      promote(block.content)
    end
  end
end

return {
  Pandoc = function(doc)
    promote(doc.blocks)
    return doc
  end,
}
