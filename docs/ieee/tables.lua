-- Pandoc filter for IEEEtran: tables -> tabularx (table* when wide), figures -> sized includegraphics.
local captions = {}
local n = 0
function Meta(m)
  for _, c in ipairs(m.tablecaptions or {}) do
    table.insert(captions, (pandoc.write(pandoc.Pandoc({ pandoc.Plain(c) }), 'latex'):gsub('%s+$', '')))
  end
end

local function cell_tex(cell)
  local s = pandoc.write(pandoc.Pandoc(cell.contents), 'latex')
  s = s:gsub('%s+$', ''):gsub('\n\n', ' '):gsub('/', '/\\allowbreak{}'):gsub('\\_', '\\_\\allowbreak{}')
  return s
end

function Table(tbl)
  n = n + 1
  local rows, chars, ncols = {}, 0, #tbl.colspecs
  local function add(r, bold)
    local cells = {}
    for _, c in ipairs(r.cells) do
      local t = cell_tex(c)
      chars = chars + #pandoc.utils.stringify(c.contents)
      cells[#cells + 1] = bold and ('\\textbf{' .. t .. '}') or t
    end
    rows[#rows + 1] = table.concat(cells, ' & ') .. ' \\\\'
  end
  for _, r in ipairs(tbl.head.rows) do add(r, true) end
  local nhead = #rows
  for _, b in ipairs(tbl.bodies) do for _, r in ipairs(b.body) do add(r, false) end end
  local wide = ncols >= 5 or chars > 450
  local env = wide and 'table*' or 'table'
  local width = wide and '\\textwidth' or '\\columnwidth'
  local spec = string.rep('>{\\raggedright\\arraybackslash}X', ncols)
  local out = { '\\begin{' .. env .. '}[!t]', '\\centering\\footnotesize',
    '\\caption{' .. (captions[n] or '') .. '}', '\\label{tab:' .. n .. '}',
    '\\renewcommand{\\arraystretch}{1.15}',
    '\\begin{tabularx}{' .. width .. '}{' .. spec .. '}', '\\toprule' }
  for i, r in ipairs(rows) do
    out[#out + 1] = r
    if i == nhead then out[#out + 1] = '\\midrule' end
  end
  out[#out + 1] = '\\bottomrule'
  out[#out + 1] = '\\end{tabularx}'
  out[#out + 1] = '\\end{' .. env .. '}'
  return pandoc.RawBlock('latex', table.concat(out, '\n'))
end

function Figure(fig)
  local img
  fig:walk({ Image = function(i) img = i end })
  local wide = img.src:match('architecture') ~= nil
  local env = wide and 'figure*' or 'figure'
  local width = wide and '0.9\\textwidth' or '\\columnwidth'
  local cap = pandoc.write(pandoc.Pandoc({ pandoc.Plain(img.caption) }), 'latex'):gsub('%s+$', '')
  cap = cap:gsub('^Figure %d+%. ', '')
  local label = img.src:match('(fig%d)')
  return pandoc.RawBlock('latex', table.concat({ '\\begin{' .. env .. '}[!t]', '\\centering',
    '\\includegraphics[width=' .. width .. ']{../' .. img.src .. '}',
    '\\caption{' .. cap .. '}', '\\label{' .. label .. '}', '\\end{' .. env .. '}' }, '\n'))
end

return { { Meta = Meta }, { Table = Table, Figure = Figure } }
