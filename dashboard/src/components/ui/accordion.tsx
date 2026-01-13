import * as React from "react"
import { ChevronDown } from "lucide-react"
import { cn } from "@/lib/utils"

interface AccordionContextValue {
  openItems: Set<string>
  toggleItem: (value: string) => void
  type: "single" | "multiple"
}

const AccordionContext = React.createContext<AccordionContextValue | undefined>(undefined)

function useAccordionContext() {
  const context = React.useContext(AccordionContext)
  if (!context) {
    throw new Error("useAccordionContext must be used within an Accordion component")
  }
  return context
}

interface AccordionProps extends React.HTMLAttributes<HTMLDivElement> {
  type?: "single" | "multiple"
  defaultValue?: string | string[]
  value?: string | string[]
  onValueChange?: (value: string | string[]) => void
}

const Accordion = React.forwardRef<HTMLDivElement, AccordionProps>(
  ({ className, type = "single", defaultValue, value, onValueChange, children, ...props }, ref) => {
    const [openItems, setOpenItems] = React.useState<Set<string>>(() => {
      const initial = value ?? defaultValue
      if (!initial) return new Set()
      return new Set(Array.isArray(initial) ? initial : [initial])
    })

    const toggleItem = React.useCallback((itemValue: string) => {
      setOpenItems((prev) => {
        const newSet = new Set(prev)
        if (type === "single") {
          if (newSet.has(itemValue)) {
            newSet.clear()
          } else {
            newSet.clear()
            newSet.add(itemValue)
          }
        } else {
          if (newSet.has(itemValue)) {
            newSet.delete(itemValue)
          } else {
            newSet.add(itemValue)
          }
        }

        const newValue = type === "single"
          ? (newSet.size > 0 ? Array.from(newSet)[0] : "")
          : Array.from(newSet)
        onValueChange?.(newValue)

        return newSet
      })
    }, [type, onValueChange])

    return (
      <AccordionContext.Provider value={{ openItems, toggleItem, type }}>
        <div ref={ref} className={cn("w-full", className)} {...props}>
          {children}
        </div>
      </AccordionContext.Provider>
    )
  }
)
Accordion.displayName = "Accordion"

interface AccordionItemProps extends React.HTMLAttributes<HTMLDivElement> {
  value: string
}

const AccordionItem = React.forwardRef<HTMLDivElement, AccordionItemProps>(
  ({ className, value, children, ...props }, ref) => {
    return (
      <div
        ref={ref}
        data-value={value}
        className={cn("border-b", className)}
        {...props}
      >
        {React.Children.map(children, (child) => {
          if (React.isValidElement(child)) {
            return React.cloneElement(child as React.ReactElement<{ itemValue?: string }>, { itemValue: value })
          }
          return child
        })}
      </div>
    )
  }
)
AccordionItem.displayName = "AccordionItem"

interface AccordionTriggerProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  itemValue?: string
}

const AccordionTrigger = React.forwardRef<HTMLButtonElement, AccordionTriggerProps>(
  ({ className, children, itemValue, ...props }, ref) => {
    const { openItems, toggleItem } = useAccordionContext()
    const isOpen = itemValue ? openItems.has(itemValue) : false

    return (
      <button
        ref={ref}
        type="button"
        className={cn(
          "flex flex-1 items-center justify-between py-4 font-medium transition-all hover:underline [&[data-state=open]>svg]:rotate-180",
          className
        )}
        onClick={() => itemValue && toggleItem(itemValue)}
        data-state={isOpen ? "open" : "closed"}
        {...props}
      >
        {children}
        <ChevronDown className="h-4 w-4 shrink-0 transition-transform duration-200" />
      </button>
    )
  }
)
AccordionTrigger.displayName = "AccordionTrigger"

interface AccordionContentProps extends React.HTMLAttributes<HTMLDivElement> {
  itemValue?: string
}

const AccordionContent = React.forwardRef<HTMLDivElement, AccordionContentProps>(
  ({ className, children, itemValue, ...props }, ref) => {
    const { openItems } = useAccordionContext()
    const isOpen = itemValue ? openItems.has(itemValue) : false

    if (!isOpen) return null

    return (
      <div
        ref={ref}
        className={cn("overflow-hidden text-sm transition-all", className)}
        {...props}
      >
        <div className="pb-4 pt-0">{children}</div>
      </div>
    )
  }
)
AccordionContent.displayName = "AccordionContent"

export { Accordion, AccordionItem, AccordionTrigger, AccordionContent }
