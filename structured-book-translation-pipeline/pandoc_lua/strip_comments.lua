-- P08 Lua filter 3/3 (strip_comments)
--
-- Mirrors the page/NOTRANSLATE comment-removal regexes in publish.py
-- `clean_for_latex` (regexes #2-#4) at the AST level:
--
--   <!-- PAGE_JSON: NNNN -->   /  <!-- PAGE: NNNN -->
--   <!-- NOTRANSLATE:BEGIN ... -->   /  <!-- NOTRANSLATE:END -->
--
-- Legacy parity: only PAGE_JSON/PAGE and NOTRANSLATE markers are dropped.
-- <!-- IMAGE ... --> comments (and any other HTML comments) are preserved
-- and are subsequently ignored by Pandoc's LaTeX writer exactly as in the
-- legacy path, where only the two marker families were text-stripped.

local patterns = {
  "^%s*<!%-%-%s*PAGE_JSON:%s*%d+%s*%-%->%s*$",
  "^%s*<!%-%-%s*PAGE:%s*%d+%s*%-%->%s*$",
  "^%s*<!%-%-%s*NOTRANSLATE:BEGIN[^>]*%-%->%s*$",
  "^%s*<!%-%-%s*NOTRANSLATE:END%s*%-%->%s*$",
}

local function should_strip(text)
  for _, pattern in ipairs(patterns) do
    if text:match(pattern) then
      return true
    end
  end
  return false
end

return {
  RawBlock = function(element)
    if element.format == "html" and should_strip(element.text) then
      return {}
    end
  end,
  RawInline = function(element)
    if element.format == "html" and should_strip(element.text) then
      return {}
    end
  end,
}
