-- P08 Lua filter 1/3 (drop_title_page)
--
-- Mirrors publish.py `_strip_rendered_title_page` (regex #12) at the AST
-- level for the Lua-filter TeX path: render.py writes the book H1 (title)
-- and author line into book_translated.md, but book.tex owns the title
-- page, so the first H1 whose text equals --metadata title is dropped,
-- together with the author paragraph that directly follows it.
--
-- Later/other H1 headings are left untouched (stateful `dropped` flag),
-- matching the legacy text path which only strips the very first title
-- block. Plain-text comparison mirrors the legacy string equality check.

local function plain(value)
  if value == nil then
    return ""
  end
  if type(value) == "string" then
    return value
  end
  if value.t == "MetaString" then
    return value.text or ""
  end
  local ok, result = pcall(pandoc.utils.stringify, value)
  if ok and result then
    return result
  end
  return ""
end

local function drop_title(doc)
  local title = plain(doc.meta["title"])
  if title == "" then
    return doc
  end
  local author = plain(doc.meta["author"])
  local blocks = doc.blocks
  local dropped = false
  local index = 1
  while index <= #blocks do
    local block = blocks[index]
    if
      not dropped
      and block.t == "Header"
      and block.level == 1
      and plain(block.content) == title
    then
      table.remove(blocks, index)
      dropped = true
      if author ~= "" and blocks[index] and blocks[index].t == "Para" and plain(blocks[index].content) == author then
        table.remove(blocks, index)
      end
    else
      index = index + 1
    end
  end
  return doc
end

return {
  Pandoc = function(doc)
    return drop_title(doc)
  end,
}
