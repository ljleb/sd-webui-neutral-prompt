import functools


class ModuleHijacker:
    def __init__(self, module):
        self.__module = module
        self.__original_functions = dict()

    def hijack(self, attribute):
        if attribute not in self.__original_functions:
            self.__original_functions[attribute] = getattr(self.__module, attribute)

        def decorator(function):
            setattr(self.__module, attribute, functools.partial(function, original_function=self.__original_functions[attribute]))
            return function

        return decorator

    def reset_module(self):
        for attribute, original_function in self.__original_functions.items():
            setattr(self.__module, attribute, original_function)

        self.__original_functions.clear()

    @staticmethod
    def install_or_get(module, hijacker_attribute, on_uninstall=lambda _callback: None):
        if not hasattr(module, hijacker_attribute):
            module_hijacker = ModuleHijacker(module)
            setattr(module, hijacker_attribute, module_hijacker)
            on_uninstall(lambda: delattr(module, hijacker_attribute))
            on_uninstall(module_hijacker.reset_module)
            return module_hijacker
        else:
            return getattr(module, hijacker_attribute)
